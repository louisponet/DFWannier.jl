div1(x, y) = div(x - 1, y) + 1

w_eachindex(m::Matrix) = eachindex(m)

struct TbBlock{T <: AbstractFloat, M <: AbstractMatrix{Complex{T}}, MI<:AbstractMatrix{Int}, VS <: Matrix{Vector{Vec3{Int}}}, LT<:Length{T}}
    R_cart  ::Vec3{LT}
    R_cryst ::Vec3{Int}
    wigner_seitz_shifts_cryst::VS
    wigner_seitz_shifts_cart::Matrix{Vector{Vec3{LT}}}
    # Like w90 irdist_ws: The integer number of unit cells to shift the Wannier function j to put its center inside the wigner-seitz of wannier function i. Can have multiple equivalent shifts (maximum of 8), they are all stored. 
    wigner_seitz_nshifts::MI
    wigner_seitz_degeneracy::Int #not sure if we need to keep this
    # For example on boundaries of the supercell
    block   ::M
end

block(x::TbBlock) = x.block

for f in (:getindex, :size, :similar)
	@eval Base.$f(h::TbBlock, args...) = $f(block(h), args...)
end

LinearAlgebra.eigen(h::TbBlock) =
	eigen(block(h))

const TbHami{T, M, MI, VS, LT}  = Vector{TbBlock{T, M, MI, VS, LT}}

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
	@eval $op(t::TbBlock{T}, v::T) where {T} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), v))
	@eval $op(v::T, t::TbBlock{T}) where {T} = TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)))
	@eval $op(t::TbBlock{T,M}, v::M) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), v))
	@eval $op(v::M, t::TbBlock{T,M}) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)))
	@eval $op(t::TbBlock{T,M}, v::TbBlock{T,M}) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), block(v)))
end

function Hk!(out::AbstractMatrix, tbhami::TbHami, kpoint::Vec3)
    fill!(out, zero(eltype(out)))
    fourier_transform(tbhami, kpoint) do i, iR, R_cart, b, fac
        @inbounds out[i] += fac * block(b)[i]
    end
end

function Hk(tbhami::TbHami, kpoint::Vec3)
    out = similar(tbhami[1].block)
    Hk!(out, tbhami, kpoint)
    return out
end

"Fourier transforms the tight binding hamiltonian and calls the R_function with the current index and the phase."
function fourier_transform(R_function::Function, tb_hami::TbHami{T}, kpoint::Vec3) where {T}
    for (iR, b) in enumerate(tb_hami)
        degen = b.wigner_seitz_degeneracy
        shifts_used = 0
        for i in eachindex(block(b))
            # n_shifts = b.wigner_seitz_nshifts[i]
            # for is in 1:n_shifts
            #     shift = b.wigner_seitz_shifts_cryst[shifts_used + is]
            #     R_cryst = b.R_cryst + shift
            #     fac = ℯ^(2im*π*(R_cryst ⋅ kpoint))/(degen * n_shifts)
            #     R_function(i, iR, b.R_cart + b.wigner_seitz_shifts_cart[shifts_used+is],  b, fac)
            # end
            # shifts_used += n_shifts

            cart_shifts  = b.wigner_seitz_shifts_cart[i]
            cryst_shifts = b.wigner_seitz_shifts_cryst[i]
            # n_shifts     = b.wigner_seitz_nshifts[i]
            n_shifts     = length(cart_shifts)
            for is in 1:n_shifts
                shift = cryst_shifts[is]
                R_cryst = b.R_cryst + shift
                fac = ℯ^(2im*π*(R_cryst ⋅ kpoint))/(degen * n_shifts)
                R_function(i, iR, b.R_cart + cart_shifts[is],  b, fac)
            end
            shifts_used += n_shifts
        end
    end
end

"Fourier transforms the tight binding hamiltonian and calls the R_function with the current index and the phase."
function fourier_transform_nows(R_function::Function, tb_hami::TbHami{T}, kpoint::Vec3) where {T}
    for b in tb_hami
        degen = b.wigner_seitz_degeneracy
        fac = ℯ^(2im*π*(b.R_cryst ⋅ kpoint))/degen
        for i in eachindex(block(b))
            R_function(i, b, fac)
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
@with_kw mutable struct WannierBand{T<:AbstractFloat} <: Band
    kpoints_cryst ::Vector{Vec3{T}}
    eigvals        ::Vector{T}
    eigvec         ::Vector{Vector{Complex{T}}}
    cms      ::Vector{Point3{T}} = Point3{T}[]
    angmoms  ::Vector{Vector{Point3{T}}} = Vector{Point3{T}}[]
    spins    ::Vector{Vector{Point3{T}}} = Vector{Point3{T}}[]
end

function WannierBand(kpoints::Vector{Vec3{T}}, dim::Int) where T
    klen = length(kpoints)
    WannierBand{T}(kpoints_cryst=kpoints, eigvals=zeros(T, klen), eigvec=[zeros(Complex{T}, dim) for k=1:klen])
end

wannierbands(kpoints::Vector{<:Vec3}, dim::Int) =
	[WannierBand(kpoints, dim) for i=1:dim]

function wannierbands(tbhamis::TbHami, kpoints::Vector{<:Vec3})
    matdim = blocksize(tbhamis, 2)
    outbands = wannierbands(kpoints, matdim)
	calc_caches = [EigCache(block(tbhamis[1])) for i = 1:nthreads()]
    for i = 1:length(kpoints)
	    tid  = threadid()
	    k    = kpoints[i]
	    c    = calc_caches[tid]
        hami = Hk(tbhamis, k)
        eigvals, eigvecs = eigen(hami, c)
        for e=1:length(eigvals)
            outbands[e].eigvals[i] = eigvals[e]
            outbands[e].eigvec[i] = eigvecs[:,e]
            outbands[e].kpoints_cryst[i] = k
        end
    end
    return outbands
end
wannierbands(tbhamis, dfbands::Vector{<:DFBand}) =
	wannierbands(tbhamis, dfbands[1].k_points_cryst)

function character_contribution(wband::WannierBand, atoms::Vector{<:AbstractAtom})
    contributions = zeros(length(wband.kpoints_cryst))
    for (i, v) in enumerate(wband.eigvec)
        for a in atoms
            contributions[i] += real(sum(conj.(v[a]) .* v[a]))
        end
    end
    return contributions
end

