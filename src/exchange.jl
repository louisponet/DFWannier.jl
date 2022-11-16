import DFControl.Structures: Orbital, Structure, orbital, size

import DFControl.Display.Crayons: @crayon_str
import DFControl.Display: dfprintln, dfprint

struct SiteDiagonalD{T<:AbstractFloat}
	values ::Vector{T}
	T      ::Matrix{Complex{T}}
end


function setup_ω_grid(ωh, ωv, n_ωh, n_ωv, offset = 0.001)
    return vcat(range(ωh, ωh + ωv * 1im; length = n_ωv)[1:end-1],
                range(ωh + ωv * 1im, offset + ωv * 1im; length = n_ωh)[1:end-1],
                range(offset + ωv * 1im, offset; length = n_ωv))
end

struct ExchangeKGrid{T,MT} <: AbstractKGrid{T}
    hamiltonian_kgrid::HamiltonianKGrid{T,MT}
    phases::Vector{Complex{T}}
    D::Matrix{Complex{T}}
end

core_kgrid(x::ExchangeKGrid) = core_kgrid(x.hamiltonian_kgrid)
eigvecs(x::ExchangeKGrid) = eigvecs(x.hamiltonian_kgrid)
eigvals(x::ExchangeKGrid) = eigvals(x.hamiltonian_kgrid)

function ExchangeKGrid(hami::TBHamiltonian, kpoints::Vector{Vec3{T}}, R = zero(Vec3{T})) where {T}
    D = ThreadCache(zeros_block(hami))
    hami_kpoints = HamiltonianKGrid(hami, kpoints, x -> D .+= x)
    nk = length(hami_kpoints)
    Ds = reduce(+, D)
    return ExchangeKGrid(hami_kpoints, phases(kpoints, R),
                         Array(Ds[Up()] - Ds[Down()]) / nk)
end

function calc_greens_functions(ω_grid, kpoints, μ::T) where {T}
    g_caches = [ThreadCache(fill!(similar(eigvecs(kpoints)[1]), zero(Complex{T})))
                for i in 1:3]
    Gs = [fill!(similar(eigvecs(kpoints)[1]), zero(Complex{T})) for i in 1:length(ω_grid)]
    function iGk!(G, ω)
        fill!(G, zero(Complex{T}))
        return integrate_Gk!(G, ω, μ, kpoints, cache.(g_caches))
    end
    p = Progress(length(ω_grid), 1, "Calculating contour G(ω)...")
    @threads for j in 1:length(ω_grid)
        iGk!(Gs[j], ω_grid[j])
        next!(p)
    end
    return Gs
end

function integrate_Gk!(G::AbstractMatrix, ω, μ, kpoints, caches)
    dim = blockdim(G)
    cache1, cache2, cache3 = caches

    @inbounds for ik in 1:length(kpoints)
        # Fill here needs to be done because cache1 gets reused for the final result too
        fill!(cache1, zero(eltype(cache1)))
        for x in 1:2dim
            cache1[x, x] = 1.0 / (μ + ω - eigvals(kpoints)[ik][x])
        end
        # Basically Hvecs[ik] * 1/(ω - eigvals[ik]) * Hvecs[ik]'

        mul!(cache2,     eigvecs(kpoints)[ik], cache1)
        adjoint!(cache3, eigvecs(kpoints)[ik])
        mul!(cache1,     cache2, cache3)
        t = kpoints.phases[ik]
        tp = t'
        for i in 1:dim, j in 1:dim
            G[i, j]         += cache1[i, j] * t
            G[i+dim, j+dim] += cache1[i+dim, j+dim] * tp
            G[i+dim, j]      = cache1[i+dim, j]
            G[i, j+dim]      = cache1[i, j+dim]
        end
    end
    return G ./= length(kpoints)
end

function integrate_Gk!(G::ColinMatrix, ω, μ, kpoints, caches)
    dim = size(G, 1)
    cache1, cache2, cache3 = caches

    @inbounds for ik in 1:length(kpoints)
        # Fill here needs to be done because cache1 gets reused for the final result too
        fill!(cache1, zero(eltype(cache1)))
        for x in 1:dim
            cache1[x, x]     = 1.0 / (μ + ω - kpoints.hamiltonian_kgrid.eigvals[ik][x])
            cache1[x, x+dim] = 1.0 / (μ + ω - kpoints.hamiltonian_kgrid.eigvals[ik][x+dim])
        end
        # Basically Hvecs[ik] * 1/(ω - eigvals[ik]) * Hvecs[ik]'

        mul!(cache2, kpoints.hamiltonian_kgrid.eigvecs[ik], cache1)
        adjoint!(cache3, kpoints.hamiltonian_kgrid.eigvecs[ik])
        mul!(cache1, cache2, cache3)
        t = kpoints.phases[ik]
        tp = t'
        for i in 1:dim, j in 1:dim
            G[i, j]     += cache1[i, j] * t
            G[i, j+dim] += cache1[i, j+dim] * tp
        end
    end
    return G ./= length(kpoints)
end

function integrate_Gk!(G_forward::ThreadCache, G_backward::ThreadCache, ω, μ, Hvecs, Hvals,
                       R, kgrid, caches)
    dim = size(G_forward, 1)
    cache1, cache2, cache3 = caches

    @inbounds for ik in 1:length(kgrid)
        # Fill here needs to be done because cache1 gets reused for the final result too
        fill!(cache1, zero(eltype(cache1)))
        for x in 1:dim
            cache1[x, x] = 1.0 / (μ + ω - Hvals[ik][x])
        end
        # Basically Hvecs[ik] * 1/(ω - eigvals[ik]) * Hvecs[ik]'
        mul!(cache2, Hvecs[ik], cache1)
        adjoint!(cache3, Hvecs[ik])
        mul!(cache1, cache2, cache3)
        t = exp(2im * π * dot(R, kgrid[ik]))
        
        G_forward  .+= cache1 .* t
        G_backward .+= cache1 .* t'
    end
    G_forward.caches ./= length(kgrid)
    return G_backward.caches ./= length(kgrid)
end

abstract type Exchange{T<:AbstractFloat} end
Base.eltype(::Exchange{T}) where {T} = T
Base.eltype(::Type{Exchange{T}}) where {T} = T

function (::Type{E})(at1::Atom, at2::Atom; site_diagonal::Bool = false) where {E<:Exchange}
    l1 = length(uprange(at1))
    l2 = length(uprange(at2))
    return site_diagonal ? E(zeros(Float64, l1, l2), at1, at2) :
           E(zeros(Float64, l1, l1), at1, at2)
end

"""
    Exchange2ndOrder{T <: AbstractFloat}

This holds the exhanges between different orbitals and calculated sites.
Projections and atom datablocks are to be found in the corresponding wannier input file.
It turns out the ordering is first projections, then atom order in the atoms datablock.
"""
mutable struct Exchange2ndOrder{T<:AbstractFloat} <: Exchange{T}
    J::Matrix{T}
    atom1::Atom
    atom2::Atom
end

function Base.show(io::IO, e::Exchange)
    dfprint(io, crayon"red", "atom1:", crayon"reset")
    dfprintln(io, "name: $(e.atom1.name), pos: $(e.atom1.position_cryst)")
    dfprint(io, crayon"red", " atom2:", crayon"reset")
    dfprintln(io, "name: $(e.atom2.name), pos: $(e.atom2.position_cryst)")

    return dfprint(io, crayon"red", " J: ", crayon"reset", "$(tr(e.J))")
end

"""
    Exchange4thOrder{T <: AbstractFloat}

This holds the exhanges between different orbitals and calculated sites.
Projections and atom datablocks are to be found in the corresponding wannier input file.
It turns out the ordering is first projections, then atom order in the atoms datablock.
"""
mutable struct Exchange4thOrder{T<:AbstractFloat} <: Exchange{T}
    J::Matrix{T}
    atom1::Atom
    atom2::Atom
end

"""
    calc_exchanges(hamiltonian::TBHamiltonian, atoms::Vector{<:Atom}, fermi, exchange_type; kwargs...)

Calculates the magnetic exchange parameters between the `atoms`. `exchange_type` can be [`Exchange2ndOrder`](@ref) or [`Exchange4thOrder`](@ref). The `kwargs` control various numerical parameters for the calculation:
- `nk = (10,10,10)`: the amount of _k_-points to be used for the uniform interpolation grid.
- `R = (0,0,0)`: the unit cell index to which the exchange parameters are calculated.
- `ωh = -30.0`: the lower bound of the energy integration
- `ωv = 0.15`: the height of the contour in complex space to integrate the Green's functions
- `n_ωh = 3000`: number of integration points along the horizontal contour direction
- `n_ωv = 500`: number of integration points along the vertical contour direction
- `site_diagonal = false`: if `true` the hamiltonians and `Δ` will diagonalized on-site and the
returned exchange matrices hold the exchanges between well-defined orbitals. If this is not done,
the exchange matrices entries don't mean anything on themselves and a trace should be performed to
find the exchange between the spins on sites `i` and `j`.
""" 
function calc_exchanges(hami, atoms::Vector{<:Atom}, fermi::T, ::Type{E} = Exchange2ndOrder;
                        nk::NTuple{3,Int} = (10, 10, 10),
                        R = Vec3(0, 0, 0),
                        ωh::T = T(-30.0), # starting energy
                        ωv::T = T(0.15), # height of vertical contour
                        n_ωh::Int = 3000,
                        n_ωv::Int = 500,
                        temp::T = T(0.01),
                        site_diagonal = false) where {T<:AbstractFloat,E<:Exchange}
    R_     = Vec3(R...)
    μ      = fermi
    ω_grid = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)

    exchanges = E{T}[]
    for at1 in atoms
        for at2 in atoms
            push!(exchanges, E(at1, at2; site_diagonal = site_diagonal))
        end
    end
    kpoints = ExchangeKGrid(hami, uniform_shifted_kgrid(nk...), R_)

    D_ = site_diagonal ? site_diagonalize(kpoints.D, atoms) : kpoints.D
    calc_exchanges!(exchanges, μ, ω_grid, kpoints, D_)

    return exchanges
end

function site_diagonalize(D::Matrix{Complex{T}}, ats::Vector{DFC.Structures.Atom}) where {T}
    Ts = zeros(D)
    Dvals = zeros(T, size(D, 1))
    for at in ats
        t_vals, t_vecs    = eigen(Hermitian(D[at]))
        Ts[at]           .= t_vecs
        Dvals[range(at)] .= real.(t_vals)
    end
    return SiteDiagonalD(Dvals, Ts)
end

function calc_exchanges!(exchanges::Vector{<:Exchange{T}},
                         μ::T,
                         ω_grid::AbstractArray{Complex{T}},
                         kpoints,
                         D::Union{Matrix{Complex{T}},SiteDiagonalD{T}}) where {T<:AbstractFloat}
    dim = size(kpoints.hamiltonian_kgrid.eigvecs[1])
    d2 = div(dim[1], 2)
    J_caches = [ThreadCache(zeros(T, size(e.J))) for e in exchanges]
    Gs = calc_greens_functions(ω_grid, kpoints, μ)
    @threads for i in 1:length(Gs)-1
        for (eid, exch) in enumerate(exchanges)
            J_caches[eid] .+= Jω(exch, D, Gs[i], ω_grid[i+1] - ω_grid[i])
        end
    end
    for (eid, exch) in enumerate(exchanges)
        exch.J = -1e3 / 2π * reduce(+, J_caches[eid])
    end
end

spin_sign(D) = -sign(real(tr(D))) # up = +1, down = -1. If D_upup > D_dndn, onsite spin will be down and the tr(D) will be positive. Thus explaining the - in front of this.
spin_sign(D::Vector) = sign(real(sum(D))) # up = +1, down = -1

function perturbation_bubble(::Exchange2ndOrder, D_site1, G_forward, D_site2, G_backward)
    return D_site1 * G_forward * D_site2 * G_backward
end

function perturbation_bubble(::Exchange4thOrder, D_site1, G_forward, D_site2, G_backward)
    return D_site1 * G_forward * D_site2 * G_backward * D_site1 * G_forward * D_site2 *
           G_backward
end

@inline function Jω(exch, D, G, dω)
    if size(D, 1) < size(G, 1)
        ra1 = uprange(exch.atom1)
        ra2 = uprange(exch.atom2)
    else
        ra1 = range(exch.atom1)
        ra2 = range(exch.atom2)
    end
    D_site1    = view(D, ra1, ra1)
    D_site2    = view(D, ra2, ra2)
    G_forward  = view(G, exch.atom1, exch.atom2, Down())
    G_backward = view(G, exch.atom2, exch.atom1, Up())
    return spin_sign(D_site1) .* spin_sign(D_site2) .*
           imag.(perturbation_bubble(exch,
                                     D_site1,
                                     G_forward,
                                     D_site2,
                                     G_backward) * dω)
end

@inline function Jω(exch, D::SiteDiagonalD, G, dω)
    s1         = spin_sign(D.values[exch.atom1])
    s2         = spin_sign(D.values[exch.atom2])
    t          = zeros(exch.J)
    G_forward  = D.T[exch.atom1]' * G[exch.atom1, exch.atom2, Down()] * D.T[exch.atom2]
    G_backward = D.T[exch.atom2]' * G[exch.atom2, exch.atom1, Up()]   * D.T[exch.atom1]
    for j in 1:size(t, 2), i in 1:size(t, 1)
        t[i, j] = s1 * s2 *
                  imag(D.values[exch.atom1][i] *
                       G_forward[i, j] *
                       D.values[exch.atom2][j] *
                       G_backward[j, i] *
                       dω)
    end
    return t
end

mutable struct AnisotropicExchange2ndOrder{T<:AbstractFloat} <: Exchange{T}
    J::Matrix{Matrix{T}}
    atom1::Atom
    atom2::Atom
end

function AnisotropicExchange2ndOrder(at1::Atom, at2::Atom)
    return AnisotropicExchange2ndOrder{Float64}([zeros(length(range(at1)),
                                                       length(range(at1)))
                                                 for i in 1:3, j in 1:3], at1, at2)
end

function calc_anisotropic_exchanges(hami, atoms, fermi::T;
                                    nk::NTuple{3,Int} = (10, 10, 10),
                                    R = Vec3(0, 0, 0),
                                    ωh::T = T(-30.0), # starting energy
                                    ωv::T = T(0.1), # height of vertical contour
                                    n_ωh::Int = 3000,
                                    n_ωv::Int = 500,
                                    temp::T = T(0.01)) where {T<:AbstractFloat}
    μ         = fermi
    k_grid    = uniform_shifted_kgrid(nk...)
    ω_grid    = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    exchanges = setup_anisotropic_exchanges(atoms)

    Hvecs, Hvals, D = DHvecvals(hami, k_grid, atoms)

    calc_anisotropic_exchanges!(exchanges, μ, R, k_grid, ω_grid, Hvecs, Hvals, D)
    return exchanges
end

function calc_anisotropic_exchanges!(exchanges::Vector{AnisotropicExchange2ndOrder{T}},
                                     μ::T,
                                     R::Vec3,
                                     k_grid::AbstractArray{Vec3{T}},
                                     ω_grid::AbstractArray{Complex{T}},
                                     Hvecs::Vector{Matrix{Complex{T}}},
                                     Hvals::Vector{Vector{T}},
                                     D::Vector{Vector{Matrix{Complex{T}}}}) where {T<:AbstractFloat}
    dim = size(Hvecs[1])
    J_caches = [ThreadCache([zeros(T, size(e.J[i, j])) for i in 1:3, j in 1:3])
                for e in exchanges]
    g_caches = [ThreadCache(zeros(Complex{T}, dim)) for i in 1:3]
    G_forward, G_backward = [ThreadCache(zeros(Complex{T}, dim)) for i in 1:2]

    function iGk!(ω)
        fill!(G_forward, zero(Complex{T}))
        fill!(G_backward, zero(Complex{T}))
        return integrate_Gk!(G_forward, G_backward, ω, μ, Hvecs, Hvals, R, k_grid, g_caches)
    end

    for j in 1:length(ω_grid[1:end-1])
        ω  = ω_grid[j]
        dω = ω_grid[j+1] - ω
        iGk!(ω)
        # The two kind of ranges are needed because we calculate D only for the projections we care about 
        # whereas G is calculated from the full Hamiltonian, the is needed. 
        for (eid, exch) in enumerate(exchanges)
            rm = range(exch.atom1)
            rn = range(exch.atom2)
            for i in 1:3, j in 1:3 # x,y,z 
                J_caches[eid][i, j] .+= imag.((view(D[eid][i], 1:length(rm), 1:length(rm)) *
                                               view(G_forward, rm, rn) *
                                               view(D[eid][j], 1:length(rn), 1:length(rn)) *
                                               view(G_backward, rn, rm)) .* dω)
            end
        end
    end

    for (eid, exch) in enumerate(exchanges)
        exch.J = 1e3 / 2π * reduce(+, J_caches[eid])
    end
end

function setup_anisotropic_exchanges(atoms::Vector{Atom})
    exchanges = AnisotropicExchange2ndOrder{Float64}[]
    for (i, at1) in enumerate(atoms), at2 in atoms[i:end]
        push!(exchanges, AnisotropicExchange2ndOrder(at1, at2))
    end
    return exchanges
end

@doc raw"""
    DHvecvals(hami::TBHamiltonian{T, Matrix{T}}, k_grid::Vector{Vec3{T}}, atoms::Atom{T}) where T <: AbstractFloat

Calculates $D(k) = [H(k), J]$, $P(k)$ and $L(k)$ where $H(k) = P(k) L(k) P^{-1}(k)$.
`hami` should be the full Hamiltonian containing both spin-diagonal and off-diagonal blocks.
"""
function DHvecvals(hami, k_grid::AbstractArray{Vec3{T}},
                   atoms::Vector{Atom}) where {T<:AbstractFloat}
    nk = length(k_grid)
    Hvecs = [zeros_block(hami) for i in 1:nk]
    Hvals = [Vector{T}(undef, blocksize(hami, 1)) for i in 1:nk]
    δH_onsite = ThreadCache([[zeros(Complex{T}, 2length(range(at)), 2length(range(at)))
                              for i in 1:3] for at in atoms])
    calc_caches = [HermitianEigenWs(block(hami[1])) for i in 1:nthreads()]
    for i in 1:nk
        # for i=1:nk
        tid = threadid()
        # Hvecs[i] is used as a temporary cache to store H(k) in. Since we
        # don't need H(k) but only Hvecs etc, this is ok.
        Hk!(Hvecs[i], hami, k_grid[i])

        for (δh, at) in zip(δH_onsite, atoms)
            rat = range(at)
            lr  = length(rat)
            δh .+= commutator.((view(Hvecs[i], rat, rat),), at[:operator_block].J) # in reality this should be just range(at)
            # δh .+= commutator.(([Hvecs[i][rat, rat] zeros(Complex{T},lr, lr); zeros(Complex{T}, lr, lr) Hvecs[i][div(blocksize(hami, 1), 2) .+ rat, div(blocksize(hami, 1), 2) .+ rat]],), at[:operator_block].J) #in reality this should be just range(at)
        end
        eigen!(Hvals[i], Hvecs[i], calc_caches[tid])
    end
    return Hvecs, Hvals, reduce(+, δH_onsite) ./ nk
end

commutator(A1, A2) where {T} = A1 * A2 - A2 * A1

function totocc(Hvals, fermi::T, temp::T) where {T}
    totocc = zero(Complex{T})
    for i in 1:length(Hvals)
        totocc += reduce(+, 1 ./ (exp.((Hvals[i] .- fermi) ./ temp) .+ 1))
    end
    return totocc / length(Hvals)
end
