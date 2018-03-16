using DFControl: WannierDataBlock, Projection, Orbital, s, p, d, f

"""
    WannExchanges{T <: AbstractFloat}

This holds the exhanges between different orbitals and calculated sites.
Projections and atom datablocks are to be found in the corresponding wannier input file.
It turns out the ordering is first projections, then atom order in the atoms datablock.
"""
mutable struct Exchange{T <: AbstractFloat}
    J       ::Matrix{T}
    atom1   ::Atom{T}
    atom2   ::Atom{T}
    proj1   ::Projection
    proj2   ::Projection
end

function calculate_eig_totocc_D(hami_raw_up, hami_raw_dn, fermi::T, temp::T, k_grid) where T <:AbstractFloat
    totocc_t    = zeros(Complex{T}, Threads.nthreads())
    n_orb       = size(hami_from_k(hami_raw_up, k_grid[1]))[1]
    k_eigval_up = fill(Array{Complex{T}, 1}(n_orb), length(k_grid))
    k_eigvec_up = fill(Array{Complex{T}, 2}(n_orb, n_orb), length(k_grid))
    k_eigval_dn = fill(Array{Complex{T}, 1}(n_orb), length(k_grid))
    k_eigvec_dn = fill(Array{Complex{T}, 2}(n_orb, n_orb), length(k_grid))
    μ           = fermi
    D_t         = fill(zeros(Complex{T}, n_orb, n_orb), Threads.nthreads())
    # D           = zeros(Complex{T}, n_orb, n_orb)
    j=1
    for  hami in [hami_raw_up, hami_raw_dn]
        Threads.@threads for i=1:length(k_grid)
            tid = Threads.threadid()
            k = k_grid[i]
            hami_k         = hami_from_k(hami, k)
            eigval, eigvec = sorted_eig(hami_k)

            if j == 1
                k_eigval_up[i] = eigval
                k_eigvec_up[i] = eigvec
                for val in eigval
                    totocc_t[tid] += 1. / (exp((val - μ) / temp) + 1.)
                end
                D_t[tid] += hami_k
            else
                k_eigval_dn[i] = eigval
                k_eigvec_dn[i] = eigvec

                for val in eigval
                    totocc_t[tid] += 1. / (exp((val - μ) / temp) + 1.)
                end
                D_t[tid] -= hami_k
            end
        end
        j+=1
    end
    D = sum(D_t)
    totocc = sum(totocc_t)
    return k_eigval_up, k_eigval_dn, k_eigvec_up, k_eigvec_dn, totocc, D
end

function setup_exchanges(atoms::Array{Atom{T}, 1}, orbitals) where T <: AbstractFloat
    exchanges = Exchange{T}[]
    for (i, at1) in enumerate(atoms), at2 in atoms[i+1:end]
        projections1 = at1.projections
        projections2 = at2.projections
        for proj1 in projections1, proj2 in projections2
            if proj1.orb in orbitals && proj2.orb in orbitals
                push!(exchanges, Exchange{T}(zeros(T, proj1.size, proj1.size), at1, at2, proj1, proj2))
            end
        end
    end
    return exchanges
end

function setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    ω_grid = [ω - ωv * 1im for ω = ωh:abs(ωh)/n_ωh:0.]
    ω_grid = vcat(ω_grid, [ω * 1im for ω = -ωv:ωv/n_ωv:ωv/10/n_ωv])
    return ω_grid
end

function calculate_exchanges(hami_raw_up::Array, hami_raw_dn::Array,  structure::Structure, fermi::T;
                             nk::NTuple{3, Int} = (10, 10, 10),
                             R::Array{Int,1}    = [0, 0, 0],
                             ωh::T              = T(-30.), #starting energy
                             ωv::T              = T(0.5), #height of vertical contour
                             n_ωh::Int          = 300,
                             n_ωv::Int          = 50,
                             temp::T            = T(0.01),
                             orbitals::Array{Orbital, 1} = [d, f]) where T <: AbstractFloat

    @assert !isempty(structure.atoms[1].projections) "Please read a valid wannier file for structure with projections."
    μ = fermi
    atoms = structure.atoms
    k_grid = [[kx, ky, kz] for kx = 0.5/nk[1]:1/nk[1]:1, ky = 0.5/nk[2]:1/nk[2]:1, kz = 0.5/nk[3]:1/nk[3]:1]

    k_eigval_up, k_eigval_dn, k_eigvec_up, k_eigvec_dn, totocc, D =
        calculate_eig_totocc_D(hami_raw_up, hami_raw_dn, fermi, temp, k_grid)

    k_infos = [zip(k_grid, k_eigvals, k_eigvecs) for (k_eigvals, k_eigvecs) in zip([k_eigval_up, k_eigval_dn],[k_eigvec_up, k_eigvec_dn])]
    D /= prod(nk)::Int
    n_orb = size(D)[1]
    totocc /= prod(nk)::Int
    structure.data[:totocc] = real(totocc)

    ω_grid    = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    exchanges = setup_exchanges(atoms, orbitals)

    # for j=1:length(ω_grid[1:end-1])
    t_js = [fill(similar(e.J), Threads.nthreads()) for e in exchanges]
    Threads.@threads for j=1:length(ω_grid[1:end-1])
        ω  = ω_grid[j]
        dω = ω_grid[j + 1] - ω

        g = fill(zeros(T, n_orb, n_orb), 2)
        for (ki, k_info) in enumerate(k_infos)
            sign = ki * 2 - 3 #1=-1 2=1
            for (k, vals, vecs) in k_info
                g[ki] += vecs * diagm(1. ./(μ + ω .- vals)) * vecs' * exp(-2im * π * dot(sign * R, k))
            end
        end
        tid = Threads.threadid()
        for (eid, exch) in enumerate(exchanges)
            s_m = exch.proj1.start
            l_m = exch.proj1.last
            s_n = exch.proj2.start
            l_n = exch.proj2.last
            t_js[eid][tid] += sign(real(trace(D[s_m:l_m, s_m:l_m]))) * sign(real(trace(D[s_n:l_n, s_n:l_n]))) * imag(D[s_m:l_m, s_m:l_m] * g[1][s_m:l_m, s_n:l_n] * D[s_n:l_n, s_n:l_n] * g[2][s_n:l_n, s_m:l_m] * dω)
        end
    end
    for (eid, exch) in enumerate(exchanges)
        exch.J = 1e3 / (2π * prod(nk)^2) * sum(t_js[eid])
    end
    structure.data[:exchanges] = exchanges
end

function calculate_exchanges(hami_up_file::String, hami_down_file::String, wannier_input_file::String, args...; kwargs...)
    structure = read_wannier_input(wannier_input_file)[2]
    calculate_exchanges(read_hami_file(hami_up_file), read_hami_file(hami_down_file), structure, args...; kwargs...)
    return structure
end



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
        if e.atom1 == atom1 && e.atom2 == atom2
            exch += trace(e.J)
        end
    end
    return exch
end
