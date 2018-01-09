using DFControl: WannierDataBlock

@enum Orbital s p d f 
function Orbital(s::Symbol)
    t = 0 
    while Symbol(Orbital(t)) != s
        t += 1
        if t > Int(f)
            error("Orbital $s not defined.")
        end
    end
    return t
end
orbsize(orbital::Orbital) = Int(orbital) * 2 + 1
orbsize(orbital::Symbol)  = Orbital(orbital) * 2 + 1 

struct WannProjection{T <: AbstractFloat}
    orb::Symbol
    atom::Symbol
    start::Int
    size::Int
    last::Int
    position::Point3D{T}
end

function get_wan_projections(filename::String, T=Float64)
    projections = Tuple[]
    atoms       = Tuple[]
    open(filename, "r") do f
        while !eof(f)
            line = lowercase(readline(f))
            if contains(line, "begin projections")
                line = readline(f)
                while !contains(line, "end")
                    if contains(line, "!") || line == ""
                        line = lowercase(readline(f))
                    end
                    if contains(line, "random")
                        error("Can't read the atomic info when projections are random!")
                    end
                    split_line   = DFControl.strip_split(line, ':')
                    atom         = Symbol(split_line[1])
                    _projections = [Symbol(proj) for proj in DFControl.strip_split(split_line[2], ';')]
                    push!(projections, (atom, _projections)) 
                    line = readline(f)
                end
            elseif contains(line, "begin") && contains(line, "atoms")
                line = readline(f)
                while !contains(line, "end")
                    if contains(line, "!")
                        line = readline(f)
                    end
                    split_line = DFControl.strip_split(line)
                    atom       = Symbol(split_line[1])
                    position   = Point3D(DFControl.parse_string_array(T, split_line[2:4]))
                    push!(atoms,(atom, position))
                    line = readline(f)
                end
            end
        end
    end

    out = Array{WannProjection, 1}(length(atoms))
    t_start = 1
    for (proj_at, projs) in projections
        for proj in projs
            for (i, (pos_at, pos)) in enumerate(atoms)
                if proj_at != pos_at
                    continue
                end
                size = orbsize(proj)
                out[i] = WannProjection(proj, pos_at, t_start, size, t_start + size - 1, pos)
                t_start += size
            end
        end
    end
    return out
end

"""
    WannExchanges{T <: AbstractFloat}

This holds the exhanges between different orbitals and calculated sites.
Projections and atom datablocks are to be found in the corresponding wannier input file.
It turns out the ordering is first projections, then atom order in the atoms datablock.
"""
struct WannExchanges{T <: AbstractFloat}
    Jmn    ::Array{Array{T,2},1}
    infos  ::Array{WannProjection,1}
    totocc ::T
end

function WannExchanges(hami_raw_up::Array, hami_raw_dn::Array,  orb_infos::Array{WannProjection,1}, fermi::T; 
                             nk::NTuple{3, Int} = (10, 10, 10),
                             R::Array{Int,1}    = [0, 0, 0],
                             ωh::T              = T(-30.), #starting energy
                             ωv::T              = T(0.5), #height of vertical contour
                             n_ωh::Int          = 300,
                             n_ωv::Int          = 50,
                             temp::T            = T(0.01)) where T <: AbstractFloat

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
                totocc += 1. / (exp((val - μ) / temp) + 1.)
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
    k_infos = [zip(k_grid, k_eigvals, k_eigvecs) for (k_eigvals, k_eigvecs) in zip([k_eigval_up, k_eigval_dn],[k_eigvec_up, k_eigvec_dn])]

    D /= prod(nk)::Int
    
    # ω_grid = [ω * 1im for ω = 0.0:ωv/n_ωv:ωv]
    # ω_grid = vcat(ω_grid, [ω + ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.])
    ω_grid = [ω + ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.]
    ω_grid = vcat(ω_grid, [ω * 1im for ω = ωv:-ωv/n_ωv:-0.6/n_ωv])

    n_orb_infos = length(orb_infos)
    Jmn = Array{Matrix{T},1}()
    for i = 1:n_orb_infos
        for _=i+1:n_orb_infos
            push!(Jmn, zeros(T,orb_infos[i].size, orb_infos[i].size))
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
        for m = 1:n_orb_infos
            s_m = orb_infos[m].start
            l_m = orb_infos[m].last
            for n = m + 1:n_orb_infos
                s_n = orb_infos[n].start
                l_n = orb_infos[n].last
                Threads.lock(mutex)
                Jmn[i] += imag(D[s_m:l_m, s_m:l_m] * g[1][s_m:l_m, s_n:l_n] * D[s_n:l_n, s_n:l_n] * g[2][s_n:l_n, s_m:l_m] * dω)
                Threads.unlock(mutex)
                i += 1
            end
        end
    end

    Jmn ./= 2π*prod(nk)^2
    Jmn .*= 1e3
    return WannExchanges(Jmn, orb_infos, real(totocc))
end
function WannExchanges(hami_up_file::String, hami_down_file::String, wannier_input_file::String, args...; kwargs...)
    tmp = orbs_info(wannier_input_file)
    WannExchanges(read_hami_file(hami_up_file), read_hami_file(hami_down_file), tmp, args...; kwargs...)
end

@inline function Jmn_index(atom1::Int, atom2::Int, total::Int)
    index = -1
    for i = 1:atom1 - 1 
        index += total - i - 1
    end
    index += atom2
    return index
end

exchange_between(atom1::Int, atom2::Int, exchanges::WannExchanges) =
    trace(exchanges.Jmn[Jmn_index(atom1, atom2, length(exchanges.infos))])
    
all_exchanges(exchanges::WannExchanges) = [trace(j) for j in exchanges.Jmn]

                             


