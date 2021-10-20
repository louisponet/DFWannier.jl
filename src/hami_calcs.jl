div1(x, y) = div(x - 1, y) + 1

w_eachindex(m::Matrix) = eachindex(m)

struct TbBlock{T <: AbstractFloat, LT, M <: AbstractMatrix{Complex{T}}}
    R_cryst ::Vec3{Int}
    R_cart  ::Vec3{LT}
    block::M #has contributions from all the ws shifts and divided by the degeneracies 1/(degen * nshifts), use for k-point interpolation
    tb_block::M #this is the correct tight binding block, with ws shifts applied such that entries represent hopping between closest two wfs
end

block(x::TbBlock) = x.block

for f in (:getindex, :size, :similar)
	@eval Base.$f(h::TbBlock, args...) = $f(block(h), args...)
end

LinearAlgebra.eigen(h::TbBlock) =
	eigen(block(h))

const TbHami{T, LT, M}  = Vector{TbBlock{T, LT, M}}

getindex(h::TbHami, R::Vec3{Int}) =
	getfirst(x -> x.R_cryst == R, h)

#some small type piracy?
Base.zeros(m::AbstractArray{T}) where {T} =
	fill!(similar(m), zero(T))

zeros_block(h::TbHami) =
	zeros(block(h[1]))

similar_block(h::TbHami) =
	similar(block(h[1]))

blocksize(h::TbHami, args...) =
	size(block(h[1]), args...)

for op in (:+, :-, :*, :/)
	@eval $op(t::TbBlock{T}, v::T) where {T} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), v), $op(t.tb_block, v))
	@eval $op(v::T, t::TbBlock{T}) where {T} = TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)), $op(v, t.tb_block))
	@eval $op(t::TbBlock{T,M}, v::M) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), v), $op(t.tb_block, v))
	@eval $op(v::M, t::TbBlock{T,M}) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)), $op(v, t.tb_block))
	@eval $op(t::TbBlock{T,M}, v::TbBlock{T,M}) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), block(v)), $op(t.tb_block, v.tb_block))
end

make_noncolin(tb::TbBlock) =
    TbBlock(tb.R_cryst, tb.R_cart, convert(NonColinMatrix, tb.block), convert(NonColinMatrix, tb.tb_block))

make_noncolin(tb::TbBlock{T, LT, ColinMatrix{Complex{T}, Matrix{Complex{T}}}}) where {T<:AbstractFloat, LT<:Length{T}}=
    TbBlock(tb.R_cryst, tb.R_cart, NonColinMatrix(tb.block[Up()], tb.block[Down()]), NonColinMatrix(tb.tb_block[Up()], tb.tb_block[Down()]))
    
make_noncolin(v::Vector) =
    [v[1:2:end];v[2:2:end]]

struct HamiltonianKGrid{T,MT<:AbstractMatrix{Complex{T}}, VT<:AbstractVector{T}} <: AbstractKGrid{T}
    core::CoreKGrid{T}
    Hk::Vector{MT}
    eigvals::Vector{VT}
    eigvecs::Vector{MT}
end
HamiltonianKGrid(kpoints::Vector{<:Vec3}, args...) = HamiltonianKGrid(CoreKGrid(kpoints), args...)

@doc raw"""
	HamiltonianKGrid(hami::TbHami{T}, nk, H_function_k::Function = x -> nothing) where T
	HamiltonianKGrid(hami::TbHami{T}, k_grid, H_function_k::Function = x -> nothing) where T

Takes a k grid, calculates Hk for each of them and diagonalizes. Only the eigenvectors and eigenvalues of Hk are stored,
the H_function_k function is called on the intermediate Hk. 
"""
function HamiltonianKGrid(hami::TbHami{T}, kpoints::Vector{<:Vec3}, Hk_function::Function = x -> nothing) where {T}
	# kpoints = [KPoint(k, blocksize(hami), R, zeros_block(hami)) for k in k_grid]
	n_eigvals = max(blocksize(hami)...)
	eigvals = hami[1].block isa AbstractMagneticMatrix ? [MagneticVector(zeros(T, n_eigvals)) for k in kpoints] : [zeros(T, n_eigvals) for k in kpoints]
	kgrid = HamiltonianKGrid(kpoints, [zeros_block(hami) for k in kpoints], eigvals, [zeros_block(hami) for k in kpoints])
	nk    = length(kpoints)
	calc_caches = [EigCache(block(hami[1])) for i=1:nthreads()]
	p = Progress(nk, 1, "Calculating H(k)...")
    @threads for i=1:nk
	    tid = threadid()
	    Hk!(kgrid.eigvecs[i], hami, k_cryst(kgrid)[i])
        kgrid.Hk[i] = copy(kgrid.eigvecs[i])
	    Hk_function(kgrid.Hk[i])
	    eigen!(kgrid.eigvals[i], kgrid.eigvecs[i], calc_caches[Threads.threadid()])
	    next!(p)
    end
    return kgrid
end

function Hk!(out::AbstractMatrix, tbhami::TbHami, kpoint::Vec3)
    fill!(out, zero(eltype(out)))
    fourier_transform(tbhami, kpoint) do i, iR, R_cart, b, fac
        @inbounds out[i] += fac * b.block[i]
    end
end

function Hk(tbhami::TbHami, kpoint::Vec3)
    out = similar(tbhami[1].block)
    Hk!(out, tbhami, kpoint)
    return out
end

Hk(g::HamiltonianKGrid) = g.Hk
eigvecs(g::HamiltonianKGrid) = g.eigvecs
eigvals(g::HamiltonianKGrid) = g.eigvals

"Fourier transforms the tight binding hamiltonian and calls the R_function with the current index and the phase."
function fourier_transform(R_function::Function, tb_hami::TbHami{T}, kpoint::Vec3) where {T}
    for (iR, b) in enumerate(tb_hami)
        fac = ℯ^(2im*π*(b.R_cryst ⋅ kpoint))
        for i in eachindex(block(b))
            R_function(i, iR, b.R_cart, b, fac)
        end
    end
end

# "Constructs the total spin-orbit-coupled Hamiltonian out of supplied angular momentums between the Wannier functions and uses the l_soc of the atoms."
# function construct_soc_hami(hami, structure::WanStructure{T})::Matrix{Complex{T}} where T
#     dim = getwandim(structure)
#     Lx_soc = zeros(Complex{T}, dim, dim)
#     Ly_soc = zeros(Complex{T}, dim, dim)
#     Lz_soc = zeros(Complex{T}, dim, dim)
#     i = 1
#     for at in atoms(structure)
#         soc = lsoc(at)
#         mom = angmom(at)
#         len = length(wfcs(at))-1
#         Lx_soc[i:i+len, i:i+len] = 0.5 * soc * getindex.(mom, 1)
#         Ly_soc[i:i+len, i:i+len] = 0.5 * soc * getindex.(mom, 2)
#         Lz_soc[i:i+len, i:i+len] = 0.5 * soc * getindex.(mom, 3)
#         i += len + 1
#     end
#     Lx_soc = (Lx_soc+Lx_soc')/2
#     Ly_soc = (Ly_soc+Ly_soc')/2
#     Lz_soc = (Lz_soc+Lz_soc')/2
#     out = [hami+Lz_soc Lx_soc-1im*Ly_soc;Lx_soc+1im*Ly_soc hami-Lz_soc]
#     return out
# end

"Holds all the calculated values from a wannier model."
@with_kw mutable struct WannierBand{T<:AbstractFloat, VT<:AbstractVector} <: DFC.AbstractBand
    kpoints_cryst ::Vector{Vec3{T}}
    eigvals        ::Vector{T}
    eigvec         ::Vector{VT}
    cms      ::Vector{Point3{T}} = Point3{T}[]
    angmoms  ::Vector{Vector{Point3{T}}} = Vector{Point3{T}}[]
    spins    ::Vector{Vector{Point3{T}}} = Vector{Point3{T}}[]
end

DFControl.eigvals(b::WannierBand) = b.eigvals

function wannierbands(tbhamis::TbHami{T}, kpoints::Vector{<:Vec3}) where {T}
    matdim   = blocksize(tbhamis, 2)
    kgrid    = HamiltonianKGrid(tbhamis, kpoints)
    nbnd = size(tbhamis[1].block, 2)
    evals = [zeros(length(kpoints)) for i=1:nbnd]
    evecs = [[similar(kgrid.eigvecs[1][:, 1]) for i=1:length(kpoints)] for i=1:nbnd]
    @threads for i = 1:length(kpoints)
        eigvals, eigvecs = kgrid.eigvals[i], kgrid.eigvecs[i]
        for e=1:length(eigvals)
            evals[e][i] = eigvals[e]
            evecs[e][i] = eigvecs[:,e]
        end
    end
    return [WannierBand{T, eltype(evecs[1])}(kpoints_cryst = kpoints, eigvals=evals[i], eigvec=evecs[i]) for i =1:nbnd]
end
wannierbands(tbhamis, dfbands::Vector{<:DFC.AbstractBand}) =
	wannierbands(tbhamis, dfbands[1].k_points_cryst)
wannierbands(tbhamis, dfbands::Union{NamedTuple, Tuple}) =
	wannierbands(tbhamis, dfbands[1][1].k_points_cryst)

function energy_bins(binfunc::Function, wbands::Vector{<:WannierBand}, E_range, normalize_bins=false)
    nbins   = length(E_range)-1
    bins    = zeros(typeof(binfunc(wbands[1].eigvec[1])), nbins)
    # sum = zero(typeof(binfunc(bands[1].eigvec[1])))
    nperbin = zeros(Int, nbins)
    for b in wbands
        for (e, v) in zip(b.eigvals, b.eigvec)
            ie = findfirst(i -> E_range[i] <= e <= E_range[i+1], 1:nbins)
            if ie === nothing
                continue
            end
            bins[ie] += binfunc(v)
            nperbin[ie] += 1
        end
    end
    if normalize_bins #like taking the mean
        for i=1:nbins
            if nperbin[i] > 0
                bins[i] /= nperbin[i]
            end
        end
    end
    return bins
end

function character_contribution(wband::WannierBand, atoms::Vector{Atom})
    contributions = zeros(length(wband.kpoints_cryst))
    for (i, v) in enumerate(wband.eigvec)
        for a in atoms
            contributions[i] += norm(v[a])^2
        end
    end
    return contributions
end

function DFControl.FileIO.pdos(wbands::Vector{<:WannierBand}, atoms::Vector{Atom}, dE = 0.02)
    Emin = minimum(wbands[1].eigvals)
    Emax = maximum(wbands[end].eigvals)
    E_range = range(Emin, Emax, step=dE)

    bins = energy_bins(wbands, E_range, false) do v
        tot = 0.0
        for a in atoms
            tot += norm(v[a])^2/dE
        end
        return tot
    end
    return (E=E_range, pdos=bins./length(wbands[1].kpoints_cryst))
end

kpdos(bands::Vector{<:WannierBand}, atoms::Vector{Atom}) = map(x -> character_contribution(x, atoms), bands)
