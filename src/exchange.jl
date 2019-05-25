using DFControl: Projection, Orbital, Structure, orbital, size, orbsize

uniform_shifted_kgrid(::Type{T}, nkx::Integer, nky::Integer, nkz::Integer) where {T} =
	[Vec3{T}(kx, ky, kz) for kx = 0.5/nkx:1/nkx:1, ky = 0.5/nky:1/nky:1, kz = 0.5/nkz:1/nkz:1]

uniform_shifted_kgrid(nkx::Integer, nky::Integer, nkz::Integer) = uniform_shifted_kgrid(Float64, nkx, nky, nkz)

setup_ω_grid(ωh, ωv, n_ωh, n_ωv, offset=0.001) = vcat(range(ωh,              ωh + ωv*1im,     length=n_ωv)[1:end-1],
											         range(ωh + ωv*1im,     offset + ωv*1im, length=n_ωh)[1:end-1],
											         range(offset + ωv*1im, offset,          length=n_ωv))

"""
    Exchange{T <: AbstractFloat}

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

function calc_exchanges(hami,  atoms, fermi::T;
                        nk::NTuple{3, Int} = (10, 10, 10),
                        R                  = Vec3(0, 0, 0),
                        ωh::T              = T(-30.), #starting energy
                        ωv::T              = T(0.1), #height of vertical contour
                        n_ωh::Int          = 3000,
                        n_ωv::Int          = 500,
                        temp::T            = T(0.01)) where T <: AbstractFloat

    μ               = fermi
    k_grid          = uniform_shifted_kgrid(nk...)
    ω_grid          = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    exchanges       = setup_exchanges(atoms)

    Hvecs, Hvals, D = DHvecvals(hami, k_grid)

    calc_exchanges!(exchanges, μ, R, k_grid, ω_grid, Hvecs, Hvals, D)
    return exchanges
end

function setup_exchanges(atoms::Vector{<:AbstractAtom{T}}) where T <: AbstractFloat
    exchanges = Exchange{T}[]
    for (i, at1) in enumerate(atoms), at2 in atoms[i+1:end]
        for proj1 in projections(at1), proj2 in projections(at2)
            push!(exchanges, Exchange{T}(zeros(T, orbsize(proj1), orbsize(proj1)), at1, at2, proj1, proj2))
        end
    end
    return exchanges
end


@doc raw"""
	DHvecvals(hami::TbHami{T, BandedBlockBandedMatrix{T}}, k_grid::Vector{Vec3{T}}, atoms::AbstractAtom{T}) where T <: AbstractFloat

Calculates $D(k) = [H(k), J]$, $P(k)$ and $L(k)$ where $H(k) = P(k) L(k) P^{-1}(k)$.
`hami` should be a BandedBlockBandedMatrix with $H_{up}, H_{down}$ blocks on the diagonal.
"""
function DHvecvals(hami::TbHami{T, BlockBandedMatrix{Complex{T}}}, k_grid::AbstractArray{Vec3{T}}) where T <: AbstractFloat
	dim   = blockdim(hami)[1]
	d2    = div(dim, 2)

	b_ranges = [1:d2, d2+1:dim]

	nk    = length(k_grid)
    Hvecs = [zeros_block(hami) for i=1:nk]
    Hvals = [Vector{T}(undef, dim[1]) for i=1:nk]
    D     = ThreadCache(zeros_block(hami))
	work_caches  = [zeros(T, d2) for i=1:nthreads()]
    rwork_caches = [zeros(T, 3d2-2) for i=1:nthreads()]

    lwork = 
    # @threads for i=1:length(k_grid)
    for i=1:length(k_grid)
	    tid = threadid()
        #= Hvecs[j][i] is used as a temporary cache to store H(k) in. Since we
        don't need H(k) but only Hvecs etc, this is ok.
        =#
        hvk = Hvecs[i]
        Hk!(hvk, hami, k_grid[i])
        D.caches[tid].data .+= hvk.data
        @time for j = 1:2
	        b = Block(j, j)
	        hb = view(hvk, b)
	        # syev!('V', 'U', hvk[b_ranges[1],b_ranges[1]], calc_caches[1])

	        Hvals[i][b_ranges[j]], hvk[b] = syev!('V', 'U', hb, work_caches[tid], rwork_caches[tid])
        end
    end
	Ds = sum(D)

    return Hvecs, Hvals, (Ds[Block(1,1)] - Ds[Block(2,2)])/length(k_grid)
end

function calc_exchanges!(exchanges::Vector{Exchange{T}},
	                                 μ         ::T,
	                                 R         ::Vec3,
	                                 k_grid    ::AbstractArray{Vec3{T}},
	                                 ω_grid    ::AbstractArray{Complex{T}},
	                                 Hvecs     ::Vector{BlockBandedMatrix{Complex{T}}},
	                                 Hvals     ::Vector{Vector{T}},
	                                 D         ::Matrix{Complex{T}}) where T <: AbstractFloat
    dim      = size(Hvecs[1])
    J_caches = [ThreadCache(zeros(T, size(e.J))) for e in exchanges]
    g_caches = [ThreadCache(fill!(similar(Hvecs[1]), zero(Complex{T}))) for i=1:3]
    G = ThreadCache(fill!(similar(Hvecs[1]), zero(Complex{T})))
    function iGk!(ω)
	    fill!(G, zero(Complex{T}))
        integrate_Gk!(G, ω, μ, Hvecs, Hvals, R, k_grid, g_caches)
    end

    # @threads for j=1:length(ω_grid[1:end-1])
    for j=1:length(ω_grid[1:end-1])
        ω   = ω_grid[j]
        dω  = ω_grid[j + 1] - ω
        iGk!(ω)
		# The two kind of ranges are needed because we calculate D only for the projections we care about
		# whereas G is calculated from the full Hamiltonian, the is needed.
        for (eid, exch) in enumerate(exchanges)
            rm  = range(exch.proj1)
            rn  = range(exch.proj2)
            D_rm=view(D, rm, rm)
            D_rn=view(D, rn, rn)
            J_caches[eid] .+= sign(real(tr(D_rm))) .*
                              sign(real(tr(D_rn))) .*
                              imag.(D_rm *
                                    view(G[Block(1, 1)], rm, rn) *
                                    D_rn *
                                    view(G[Block(2,2)], rn, rm) *
                                    dω)
        end
    end

    for (eid, exch) in enumerate(exchanges)
        exch.J = 1e3 / 2π * sum(J_caches[eid])
    end
end

function integrate_Gk!(G::ThreadCache{<:BlockBandedMatrix}, ω::T, μ, Hvecs, Hvals, R, kgrid, caches) where {T <: Complex}
    dim = size(G, 1)
   	dim_2 = div(dim, 2) 
	cache1, cache2, cache3 = caches

    @threads for ik=1:length(kgrid)
    # for ik=1:length(kgrid)
	    tid = threadid()
	    # Fill here needs to be done because cache1 gets reused for the final result too
        fill!(cache1, zero(T))
        for x=1:dim
            cache1[x, x] = 1.0 /(μ + ω - Hvals[ik][x])
        end
     	# Basically Hvecs[ik] * 1/(ω - eigvals[ik]) * Hvecs[ik]'
        mul!(cache2, Hvecs[ik], cache1)
        adjoint!(cache3, Hvecs[ik])
        mul!(cache1, cache2, cache3)
		t = exp(2im * π * dot(R, kgrid[ik]))
		tp = t'
		for j = 1:dim_2, k = 1:dim_2
	        G[j, k] += cache1[j, k] * t
	        G[dim_2+j, dim_2+k] += cache1[dim_2+j, dim_2+k] * tp
        end
    end
    G.caches  ./= length(kgrid)
end
# function calcexchanges(hamis,  structure::Structure, fermi::T;
#                              nk::NTuple{3, Int} = (10, 10, 10),
#                              R                  = Vec3(0, 0, 0),
#                              ωh::T              = T(-30.), #starting energy
#                              ωv::T              = T(0.1), #height of vertical contour
#                              n_ωh::Int          = 300,
#                              n_ωv::Int          = 50,
#                              temp::T            = T(0.01),
#                              orbitals::Array{Symbol, 1} = [:d, :f]) where T <: AbstractFloat
#     orbitals = orbital.(orbitals)
#     @assert !all(isempty.(projections.(DFControl.atoms(structure)))) "Please read a valid wannier file for structure with projections."
#     nth      = Threads.nthreads()
#     μ        = fermi
#     atoms    = structure.atoms
#     k_grid   = [Vec3(kx, ky, kz) for kx = 0.5/nk[1]:1/nk[1]:1, ky = 0.5/nk[2]:1/nk[2]:1, kz = 0.5/nk[3]:1/nk[3]:1]

#     Hvecs, Hvals, D = DHvecvals(hamis, k_grid)
#     n_orb = size(D, 1)

#     ω_grid    = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
#     exchanges = setup_exchanges(atoms, orbitals)

#     t_js                      = [[zeros(T, size(e.J)) for t=1:nth] for e in exchanges]
#     caches1, caches2, caches3 = [[zeros(Complex{T}, n_orb, n_orb) for t=1:nth] for i=1:3]
#     totocc_t                  = [zero(Complex{T}) for t=1:nth]
#     gs                        = [[zeros(Complex{T}, n_orb, n_orb) for n=1:2] for t  =1:nth]
#     # Threads.@threads for j=1:length(ω_grid[1:end-1])
#     Threads.@threads for j=1:length(ω_grid[1:end-1])
#         tid = Threads.threadid()
#         ω   = ω_grid[j]
#         dω  = ω_grid[j + 1] - ω
#         g   = gs[tid]
#         for s = 1:2
#             R_ = (-1)^(s-1) * R #R for spin up (-1)^(0) == 1, -R for spin down
#             G!(g[s], caches1[tid], caches2[tid], caches3[tid], ω, μ, Hvecs[s], Hvals[s], R_, k_grid)
#         end
#         for (eid, exch) in enumerate(exchanges)
#             rm = range(exch.proj1)
#             rn = range(exch.proj2)
#             t_js[eid][tid] .+= sign(real(tr(view(D, rm, rm)))) .* sign(real(tr(view(D,rn, rn)))) .* imag(view(D,rm, rm) * view(g[1],rm, rn) * view(D,rn, rn) * view(g[2],rn, rm) * dω)
#         end
#     end
#     for (eid, exch) in enumerate(exchanges)
#         exch.J = 1e3 / (2π * length(k_grid)^2) * sum(t_js[eid])
#     end
#     structure.data[:totocc] = real(totocc(Hvals, fermi, temp))
#     structure.data[:exchanges] = exchanges
# end

mutable struct AnisotropicExchange{T <: AbstractFloat}
    J       ::Matrix{Matrix{T}}
    atom1   ::Atom{T}
    atom2   ::Atom{T}
    proj1   ::Projection
    proj2   ::Projection
end

function calc_anisotropic_exchanges(hami,  atoms, fermi::T;
                             nk::NTuple{3, Int} = (10, 10, 10),
                             R                  = Vec3(0, 0, 0),
                             ωh::T              = T(-30.), #starting energy
                             ωv::T              = T(0.1), #height of vertical contour
                             n_ωh::Int          = 3000,
                             n_ωv::Int          = 500,
                             temp::T            = T(0.01)) where T <: AbstractFloat

    μ               = fermi
    k_grid          = uniform_shifted_kgrid(nk...)
    ω_grid          = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    exchanges       = setup_anisotropic_exchanges(atoms)

    Hvecs, Hvals, D = DHvecvals(hami, k_grid, atoms)

    calc_anisotropic_exchanges!(exchanges, μ, R, k_grid, ω_grid, Hvecs, Hvals, D)
    return exchanges
end

function setup_anisotropic_exchanges(atoms::Vector{<: AbstractAtom{T}}) where T <: AbstractFloat
    exchanges = AnisotropicExchange{T}[]
    for (i, at1) in enumerate(atoms), at2 in atoms[i+1:end]
        for proj1 in projections(at1), proj2 in projections(at2)
            push!(exchanges, AnisotropicExchange{T}([zeros(T, orbsize(proj1), orbsize(proj1)) for i=1:3, j=1:3],
            										at1.atom,
            										at2.atom,
            										proj1,
            										proj2))
        end
    end
    return exchanges
end

@doc raw"""
	DHvecvals(hami::TbHami{T, Matrix{T}}, k_grid::Vector{Vec3{T}}, atoms::AbstractAtom{T}) where T <: AbstractFloat


Calculates $D(k) = [H(k), J]$, $P(k)$ and $L(k)$ where $H(k) = P(k) L(k) P^{-1}(k)$.
`hami` should be the full Hamiltonian containing both spin-diagonal and off-diagonal blocks.
"""
function DHvecvals(hami::TbHami{T, Matrix{Complex{T}}}, k_grid::AbstractArray{Vec3{T}}, atoms::Vector{WanAtom{T}}) where T <: AbstractFloat
	# Get all the projections that we care about, basically the indices of the hami blocks.
	all_projections = Projection[]
	append!.((all_projections,), projections.(atoms))

	# Get all J matrices corresponding to the projections
	all_Js = Vector{Matrix{Complex{T}}}[]
	for at in atoms
		for b in at[:operator_blocks]
			push!(all_Js, b.J)
		end
	end

	nk        = length(k_grid)
    Hvecs     = [zeros_block(hami) for i=1:nk]
    Hvals     = [Vector{T}(undef, blockdim(hami)[1]) for i=1:nk]
    # Hvals     = [Vector{Complex{T}}(undef, blockdim(hami)[1]) for i=1:nk]
    δH_onsite = ThreadCache([[zeros(Complex{T}, orbsize(projection), orbsize(projection)) for i=1:3] for projection in all_projections])
	calc_caches = [EigCache(block(hami[1])) for i=1:nthreads()]
    @threads for i=1:nk
    # for i=1:nk
	    tid = threadid()
        # Hvecs[i] is used as a temporary cache to store H(k) in. Since we
        # don't need H(k) but only Hvecs etc, this is ok.
        Hk!(Hvecs[i], hami, k_grid[i])

        # For each of the dh block, proj block and J combo we have to add it to the variation of onsite hami
        for (δh, projection, j) in zip(δH_onsite, all_projections, all_Js)
	        δh .+= commutator.((view(Hvecs[i], range(projection), range(projection)),), j)
        end
        # Hvals[i], Hvecs[i] = eigen(Hvecs[i])
        eigen!(Hvals[i], Hvecs[i], calc_caches[tid])
    end
    return Hvecs, Hvals, sum(δH_onsite.caches)./nk
end

commutator(A1, A2) where T = A1*A2 - A2*A1
 
function integrate_Gk!(G_forward::ThreadCache, G_backward::ThreadCache, ω::T, μ, Hvecs, Hvals, R, kgrid, caches) where {T <: Complex}
    dim = size(G_forward, 1)
	cache1, cache2, cache3 = caches

    @threads for ik=1:length(kgrid)
	    # Fill here needs to be done because cache1 gets reused for the final result too
        fill!(cache1, zero(T))
        for x=1:dim
            cache1[x, x] = 1.0 /(μ + ω - Hvals[ik][x])
        end
     	# Basically Hvecs[ik] * 1/(ω - eigvals[ik]) * Hvecs[ik]'
        mul!(cache2, Hvecs[ik], cache1)
        adjoint!(cache3, Hvecs[ik])
        mul!(cache1, cache2, cache3)
		t = exp(2im * π * dot(R, kgrid[ik]))
        G_forward  .+= cache1 .* t
        G_backward .+= cache1 .* t'
    end
    G_forward.caches  ./= length(kgrid)
    G_backward.caches ./= length(kgrid)
end

function totocc(Hvals, fermi::T, temp::T) where T
    totocc = zero(Complex{T})
    for i = 1:length(Hvals)
        totocc += sum( 1 ./ (exp.((Hvals[i] .- fermi)./temp) .+ 1))
    end
    return totocc/length(Hvals)
end
