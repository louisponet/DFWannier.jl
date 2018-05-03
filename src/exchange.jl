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

# function calculate_eig_totocc_D(hami_raw_up, hami_raw_dn, fermi::T, temp::T, k_grid) where T <:AbstractFloat
#     totocc_t    = zeros(Complex{T}, Threads.nthreads())
#     n_orb       = size(hami_raw_up[1].block)[1]
#     k_eigval_up = fill(Vector{Complex{T}}(n_orb), length(k_grid))
#     k_eigvec_up = fill(Matrix{Complex{T}}(n_orb, n_orb), length(k_grid))
#     k_eigval_dn = fill(Vector{Complex{T}}(n_orb), length(k_grid))
#     k_eigvec_dn = fill(Matrix{Complex{T}}(n_orb, n_orb), length(k_grid))
#     μ           = fermi
#     D_t         = fill(zeros(Complex{T}, n_orb, n_orb), Threads.nthreads())
#     # D           = zeros(Complex{T}, n_orb, n_orb)
#     j=1
#     for  hami in [hami_raw_up, hami_raw_dn]
#         Threads.@threads for i=1:length(k_grid)
#             tid = Threads.threadid()
#             k = k_grid[i]
#             hami_k         = Hk(hami, k)
#             eigval, eigvec = sorted_eig(hami_k)
#
#             if j == 1
#                 k_eigval_up[i] = eigval
#                 k_eigvec_up[i] = eigvec
#                 for val in eigval
#                     totocc_t[tid] += 1. / (exp((val - μ) / temp) + 1.)
#                 end
#                 D_t[tid] += hami_k
#             else
#                 k_eigval_dn[i] = eigval
#                 k_eigvec_dn[i] = eigvec
#
#                 for val in eigval
#                     totocc_t[tid] += 1. / (exp((val - μ) / temp) + 1.)
#                 end
#                 D_t[tid] -= hami_k
#             end
#         end
#         j+=1
#     end
#     D = sum(D_t)
#     totocc = sum(totocc_t)
#     return k_eigval_up, k_eigval_dn, k_eigvec_up, k_eigvec_dn, totocc, D
# end
function DHvecvals(hamis, k_grid)
    Hvecs = [[similar(hami[1].block) for i=1:length(k_grid)] for hami in hamis]
    Hvals = [[similar(hami[1].block[:,1]) for i=1:length(k_grid)] for hami in hamis]
    D    = [zeros(hamis[1][1].block) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:length(k_grid)
        for j=1:2
            fac = (-1)^j
            tid = Threads.threadid()
            Hk!(Hvecs[j][i], hamis[j], k_grid[i])
            D[tid] .+= (-1)^j .* Hvecs[j][i]
            Hvals[j][i] = eigfact!(Hermitian(Hvecs[j][i]))[:values]
        end
    end
    return Hvecs, Hvals, sum(D)/prod(size(k_grid))
end

function setup_exchanges(atoms::Vector{Atom{T}}, orbitals) where T <: AbstractFloat
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

function Gω!(G, ω, μ, Hvec, Hval, kphase)
    dim = size(G)[1]
    for iter in eachindex(G)
        for k=1:dim
            G[iter] += Hvec[iter.I[1], k] * 1./(ω + μ - Hval[k]) * Hvec[iter.I[2], k] * kphase
        end
    end
end

function G!(G, cache1, cache2, cache3, ω::T, μ, Hvecs, Hvals, R, kgrid) where T
    dim = size(G)[1]
    for ik=1:length(kgrid)
        fill!(cache1, zero(T))
        fill!(cache2, zero(T))
        cache3 .= Hvals[ik]
        k_phase = exp(-2im * π * dot(R,kgrid[ik]))
        # totocc_t[tid] += sum(1. ./ (exp.((vals .- μ) ./ temp) .+ 1.))
        for x=1:dim
            cache1[x,x] = 1./(μ + ω - cache3[x])
        end
        @into! cache2 = Hvecs[ik] * cache1
        @into! cache1 = Hvecs[ik] * conj!(cache2)
        G .+= cache1 .* k_phase
    end
end


#DON'T FORGET HAMIS ARE UP DOWN ORDERED!!!
function calculate_exchanges(hamis,  structure::Structure, fermi::T;
                             nk::NTuple{3, Int} = (10, 10, 10),
                             R::Array{Int,1}    = [0, 0, 0],
                             ωh::T              = T(-30.), #starting energy
                             ωv::T              = T(0.5), #height of vertical contour
                             n_ωh::Int          = 30,
                             n_ωv::Int          = 5,
                             temp::T            = T(0.01),
                             orbitals::Array{Orbital, 1} = [d, f]) where T <: AbstractFloat

    @assert !isempty(structure.atoms[1].projections) "Please read a valid wannier file for structure with projections."
    nth = Threads.nthreads()
    μ = fermi
    atoms = structure.atoms
    # k_grid = [[kx, ky, kz] for kx = 0.5/nk[1]:1/nk[1]:1, ky = 0.5/nk[2]:1/nk[2]:1, kz = 0.5/nk[3]:1/nk[3]:1]
    k_grid = [Vec3(kx, ky, kz) for kx = 0.:1/nk[1]:1, ky = 0.:1/nk[2]:1, kz = 0.:1/nk[3]:1]

   Hvecs, Hvals, D = DHvecvals(hamis, k_grid)

    n_orb = size(D)[1]

    ω_grid    = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    exchanges = setup_exchanges(atoms, orbitals)

    # for j=1:length(ω_grid[1:end-1])
    t_js = [[zeros(e.J) for t=1:nth] for e in exchanges]
    caches1 = [zeros(Complex{T}, n_orb, n_orb) for t=1:nth]
    caches2 = [zeros(Complex{T}, n_orb, n_orb) for t=1:nth]
    caches3 = [zeros(Complex{T}, n_orb, n_orb) for t=1:nth]
    totocc_t = [zero(Complex{T}) for i=1:nth]
    gs = [[zeros(Complex{T}, n_orb, n_orb) for n=1:2] for t=1:nth]
    Threads.@threads for j=1:length(ω_grid[1:end-1])
        tid = Threads.threadid()
        ω  = ω_grid[j]
        dω = ω_grid[j + 1] - ω
        g = gs[tid]
        for s = 1:2
            R_ = (s * 2 - 3) * R #1=-1 2=1
            G!(g[s], caches1[tid], caches2[tid], caches3[tid], ω, μ, Hvecs[s], Hvals[s], R_, k_grid)
        end
        for (eid, exch) in enumerate(exchanges)
            rm = range(exch.proj1)
            rn = range(exch.proj2)
            t_js[eid][tid] .+= sign(real(trace(view(D, rm, rm)))) .* sign(real(trace(view(D,rn, rn)))) .* imag(view(D,rm, rm) * view(g[1],rm, rn) * view(D,rn, rn) * view(g[2],rn, rm) * dω)
        end
    end
    for (eid, exch) in enumerate(exchanges)
        exch.J = 1e3 / (2π * prod(nk)^2) * sum(t_js[eid])
    end
    structure.data[:totocc] = real(sum(totocc_t))
    structure.data[:exchanges] = exchanges
end

function calculate_exchanges(hami_up_file::String, hami_down_file::String, wannier_input_file::String, args...; kwargs...)
    structure = read_wannier_input(wannier_input_file)[2]
    calculate_exchanges(symmetrize!(read_hami_file(hami_up_file,structure), structure), symmetrize!(read_hami_file(hami_down_file, structure), structure), structure, args...; kwargs...)
    return structure
end

function calculate_exchanges(hami_up, hami_down, wannier_input_file::String, args...; kwargs...)
    structure = read_wannier_input(wannier_input_file)[2]
    calculate_exchanges(hami_up, hami_down, structure, args...; kwargs...)
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
