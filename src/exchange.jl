using DFControl: WannierDataBlock, Projection, Orbital

"""
    WannExchanges{T <: AbstractFloat}

This holds the exhanges between different orbitals and calculated sites.
Projections and atom datablocks are to be found in the corresponding wannier input file.
It turns out the ordering is first projections, then atom order in the atoms datablock.
"""
struct Exchange{T <: AbstractFloat}
    J      ::Array{T, 2}
    atom1  ::Atom{T}
    atom2  ::Atom{T}
    orb1   ::Orbital
    orb2   ::Orbital
end

function calculate_exchanges(hami_raw_up::Array, hami_raw_dn::Array,  structure::Structure, fermi::T; 
                             nk::NTuple{3, Int} = (10, 10, 10),
                             R::Array{Int,1}    = [0, 0, 0],
                             ωh::T              = T(-30.), #starting energy
                             ωv::T              = T(0.5), #height of vertical contour
                             n_ωh::Int          = 300,
                             n_ωv::Int          = 50,
                             temp::T            = T(0.01)) where T <: AbstractFloat

    @assert !isempty(structure.atoms[1].data[:projections]) "Please read a valid wannier file for structure with projections."
    atoms = structure.atoms
    k_grid      = [[kx, ky, kz] for kx = 0.5/nk[1]:1/nk[1]:1, ky = 0.5/nk[2]:1/nk[2]:1, kz = 0.5/nk[3]:1/nk[3]:1]
    n_orb       = size(hami_from_k(hami_raw_up, k_grid[1]))[1]
    k_eigval_up = fill(Array{Complex{T}}(n_orb), length(k_grid))
    k_eigvec_up = fill(Array{Complex{T}}(n_orb, n_orb), length(k_grid))
    k_eigval_dn = fill(Array{Complex{T}}(n_orb), length(k_grid))
    k_eigvec_dn = fill(Array{Complex{T}}(n_orb, n_orb), length(k_grid))
    # ekin1  = 0 #not used
    mutex = Threads.Mutex() 
    totocc = zero(Complex{T})
    μ      = fermi
    D      = zeros(Complex{Float64}, n_orb, n_orb)
    j=1
    for  hami in [hami_raw_up, hami_raw_dn]
        Threads.@threads for i=1:length(k_grid)
        # for i=1:length(k_grid)
            k = k_grid[i]
            hami_k         = hami_from_k(hami, k)
            eigval, eigvec = sorted_eig(hami_k)
            for val in eigval
                # ekin1  += eig * occ
                Threads.lock(mutex)
                totocc += 1. / (exp((val - μ) / temp) + 1.)
                Threads.unlock(mutex)
            end

            if j == 1
                k_eigval_up[i] = eigval
                k_eigvec_up[i] = eigvec
                Threads.lock(mutex)
                D             += hami_k 
                Threads.unlock(mutex)
            else
                k_eigval_dn[i] = eigval
                k_eigvec_dn[i] = eigvec
                
                Threads.lock(mutex)
                D -= hami_k
                Threads.unlock(mutex)
            end
        end
        j+=1
    end
    structure.data[:totocc] = totocc

    k_infos = [zip(k_grid, k_eigvals, k_eigvecs) for (k_eigvals, k_eigvecs) in zip([k_eigval_up, k_eigval_dn],[k_eigvec_up, k_eigvec_dn])]

    D /= prod(nk)::Int
    totocc /= prod(nk)::Int 
    # ω_grid = [ωh + ω * 1im for ω = -0.6/n_ωv:ωv/n_ωv:ωv]
    # ω_grid = vcat(ω_grid, [ω + ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.])
    ω_grid = [ω - ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.]
    ω_grid = vcat(ω_grid, [ω * 1im for ω = -ωv:ωv/n_ωv:ωv/10/n_ωv])


    exchanges = Exchange{T}[]
    for (i, at1) in atoms
        projections1 = at1.data[:projections]
        for proj1 in projections1
            for at2 in atoms[i+1:end]
                projections2 = at2.data[:projections]
                for proj2 in projections2
                    push!(exchanges, Exchange{T}(zeros(T, proj1.size, proj1.size), at1, at2, proj1.orb, proj2.orb))
                end
            end 
        end
    end
    # Jmn_threads = fill(t_Jmn, Threads.nthreads())
    # Threads.@threads for i=1:length(Jmn)
    # for j=1:length(ω_grid[1:end-1])
    Threads.@threads for j=1:length(ω_grid[1:end-1])
        ω  = ω_grid[j]
        dω = ω_grid[j + 1] - ω

        g = fill(zeros(Complex{T}, n_orb, n_orb), 2)
        for (ki, k_info) in enumerate(k_infos)
            sign = ki * 2 - 3 #1=-1 2=1
            for (k, vals, vecs) in k_info 
                g[ki] += vecs * diagm(1. ./(μ + ω .- vals)) * vecs' * exp(2im * π * dot(sign * R, k))
            end
        end
        i = 1
        for m = 1:length(atoms)
            for proj1::Projection in atoms[m].data[:projections]
                s_m = proj1.start
                l_m = proj1.last
                for n = m + 1:length(atoms)
                    for proj2::Projection in atoms[n].data[:projections]
                        s_n = proj2.start
                        l_n = proj2.last
                        Threads.lock(mutex)
                        exchanges[i].J += sign(trace(D[s_m:l_m, s_m:l_m])) * sign(trace(D[s_n:l_n, s_n:l_n])) * imag(D[s_m:l_m, s_m:l_m] * g[1][s_m:l_m, s_n:l_n] * D[s_n:l_n, s_n:l_n] * g[2][s_n:l_n, s_m:l_m] * dω)
                        Threads.unlock(mutex)
                        i += 1
                    end
                end
            end
        end
    end
    map(x->x.J, exchanges) .*= 1e3 / 2π * prod(nk)^2
    structure.data[:exchanges] = exchanges
end

function calculate_exchanges(hami_up_file::String, hami_down_file::String, wannier_input_file::String, args...; kwargs...)
    structure = read_wan_input(wannier_input_file).structure
    calculate_exchanges(read_hami_file(hami_up_file), read_hami_file(hami_down_file), structure, args...; kwargs...)
    return structure
end

# function WannExchanges(hami_raw_up::Array, hami_raw_dn::Array,  orb_infos::Array{Projection,1}, fermi::T; 
#                              nk::NTuple{3, Int} = (10, 10, 10),
#                              R::Array{Int,1}    = [0, 0, 0],
#                              ωh::T              = T(-30.), #starting energy
#                              ωv::T              = T(0.5), #height of vertical contour
#                              n_ωh::Int          = 300,
#                              n_ωv::Int          = 50,
#                              temp::T            = T(0.01)) where T <: AbstractFloat

#     k_grid      = [[kx, ky, kz] for kx = 0.5/nk[1]:1/nk[1]:1, ky = 0.5/nk[2]:1/nk[2]:1, kz = 0.5/nk[3]:1/nk[3]:1]
#     n_orb       = size(hami_from_k(hami_raw_up, k_grid[1]))[1]
#     k_eigval_up = fill(Array{Complex{T}}(n_orb), length(k_grid))
#     k_eigvec_up = fill(Array{Complex{T}}(n_orb, n_orb), length(k_grid))
#     k_eigval_dn = fill(Array{Complex{T}}(n_orb), length(k_grid))
#     k_eigvec_dn = fill(Array{Complex{T}}(n_orb, n_orb), length(k_grid))
#     # ekin1  = 0 #not used
#     mutex = Threads.Mutex() 
#     totocc = zero(Complex{T})
#     μ      = fermi
#     D      = zeros(Complex{Float64}, n_orb, n_orb)
#     j=1
#     for  hami in [hami_raw_up, hami_raw_dn]
#         Threads.@threads for i=1:length(k_grid)
#         # for i=1:length(k_grid)
#             k = k_grid[i]
#             hami_k         = hami_from_k(hami, k)
#             eigval, eigvec = sorted_eig(hami_k)
#             for val in eigval
#                 # ekin1  += eig * occ
#                 Threads.lock(mutex)
#                 totocc += 1. / (exp((val - μ) / temp) + 1.)
#                 Threads.unlock(mutex)
#             end

#             if j == 1
#                 k_eigval_up[i] = eigval
#                 k_eigvec_up[i] = eigvec
#                 Threads.lock(mutex)
#                 D             += hami_k 
#                 Threads.unlock(mutex)
#             else
#                 k_eigval_dn[i] = eigval
#                 k_eigvec_dn[i] = eigvec
                
#                 Threads.lock(mutex)
#                 D -= hami_k
#                 Threads.unlock(mutex)
#             end
#         end
#         j+=1
#     end
#     k_infos = [zip(k_grid, k_eigvals, k_eigvecs) for (k_eigvals, k_eigvecs) in zip([k_eigval_up, k_eigval_dn],[k_eigvec_up, k_eigvec_dn])]

#     D /= prod(nk)::Int
#     totocc /= prod(nk)::Int 
#     # ω_grid = [ωh + ω * 1im for ω = -0.6/n_ωv:ωv/n_ωv:ωv]
#     # ω_grid = vcat(ω_grid, [ω + ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.])
#     ω_grid = [ω - ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.]
#     ω_grid = vcat(ω_grid, [ω * 1im for ω = -ωv:ωv/n_ωv:ωv/10/n_ωv])
#     println( ω_grid[1])
#     println( ω_grid[end-1])
#     n_orb_infos = length(orb_infos)
#     Jmn = Array{Matrix{T},1}()
#     for i = 1:n_orb_infos
#         for _=i+1:n_orb_infos
#             push!(Jmn, zeros(T,orb_infos[i].size, orb_infos[i].size))
#         end
#     end
#     # Jmn_threads = fill(t_Jmn, Threads.nthreads())
#     # Threads.@threads for i=1:length(Jmn)
#     # for j=1:length(ω_grid[1:end-1])
#     Threads.@threads for j=1:length(ω_grid[1:end-1])
#         ω  = ω_grid[j]
#         dω = ω_grid[j + 1] - ω

#         g = fill(zeros(Complex{T}, n_orb, n_orb), 2)
#         for (ki, k_info) in enumerate(k_infos)
#             sign = ki * 2 - 3 #1=-1 2=1
#             for (k, vals, vecs) in k_info 
#                 g[ki] += vecs * diagm(1. ./(μ + ω .- vals)) * vecs' * exp(2im * π * dot(sign * R, k))
#             end
#         end
#         i = 1
#         for m = 1:n_orb_infos
#             s_m = orb_infos[m].start
#             l_m = orb_infos[m].last
#             for n = m + 1:n_orb_infos
#                 s_n = orb_infos[n].start
#                 l_n = orb_infos[n].last
#                 Threads.lock(mutex)
#                 Jmn[i] += sign(trace(D[s_m:l_m, s_m:l_m])) * sign(trace(D[s_n:l_n, s_n:l_n])) * imag(D[s_m:l_m, s_m:l_m] * g[1][s_m:l_m, s_n:l_n] * D[s_n:l_n, s_n:l_n] * g[2][s_n:l_n, s_m:l_m] * dω)
#                 Threads.unlock(mutex)
#                 i += 1
#             end
#         end
#     end

#     Jmn ./= 2π*prod(nk)^2
#     Jmn .*= 1e3
#     return WannExchanges(Jmn, orb_infos, real(totocc))
# end
# function WannExchanges(hami_up_file::String, hami_down_file::String, wannier_input_file::String, args...; kwargs...)
#     tmp = get_wan_projections(wannier_input_file)
#     WannExchanges(read_hami_file(hami_up_file), read_hami_file(hami_down_file), tmp, args...; kwargs...)
# end

@inline function Jmn_index(atom1::Int, atom2::Int, total::Int)
    index = -1
    for i = 1:atom1 - 1 
        index += total - i - 1
    end
    index += atom2
    return index
end

function exchange_between(atom1::Atom{T}, atom2::Atom{T}, exchanges::Array{Exchange{T}, 1}) where T <: AbstractFloat
    exch = zero(T) 
    for e in exchanges
        if e.at1 == e.at2
            exch += trace(e.J)
        end
    end
end
    
all_exchanges(exchanges::WannExchanges) = [trace(j) for j in exchanges.Jmn]

                             


