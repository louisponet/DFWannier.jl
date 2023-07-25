"""
    TBBlock

Building block for [`TBOperator`](@ref). It holds the matrix elements of the operator between
central and a shifted unit cell. Upon construction, the wigner-seitz shifts are taken into
account to create the correct matrix elements between the Wannierfunctions, stored in
`tb_block`. The `block` field is basically `tb_block` but with each element divided by
the amount of Wigner-Seitz degeneracies and shifts which speeds up later _k_-point interpolation.
"""
struct TBBlock{T<:AbstractFloat,LT,M<:AbstractMatrix{Complex{T}}}
    R_cryst  :: Vec3{Int}
    R_cart   :: Vec3{LT}
    block    :: M
    tb_block :: M
end

"""
    TBOperator

Alias for a `Vector` of [`TBBlocks`](@ref TBBlock). Indexing with `NTuple{3,Int}` or `Vec3`
is supported which allows for easily retrieving the [`TBBlock`](@ref) that corresponds
to the shifted unit cell.
Aliases: `TBHamiltonian`, `TBSpin`
"""
const TBOperator{T,LT,M} = Vector{TBBlock{T,LT,M}}
const TBHamiltonian = TBOperator
const TBSpin = TBOperator
block(x::TBBlock) = x.block

"""
    generate_TBBlocks(chk::NamedTuple, O_R::Vector)

Generates the `Vector` of [`TBBlocks`](@ref TBBlock) from the Wannier90 checkpoint info in `chk`,
and the real space operator `O_R`. This preapplies the Wigner Seitz shifts and degeneracies to
speed up _k_-point interpolation.
"""
function generate_TBBlocks(chk::NamedTuple, O_R::Vector)
    c = chk.cell'
    R_cryst, degens = chk.ws_R_cryst, chk.ws_degens
    ws_shifts, ws_nshifts = chk.ws_shifts_cryst, chk.ws_nshifts
    out = [TBBlock(R, c * R, zeros(ComplexF64, chk.n_wann, chk.n_wann),
                   zeros(ComplexF64, chk.n_wann, chk.n_wann)) for R in R_cryst]

    for (h, R, shifts, nshifts, d) in zip(O_R, R_cryst, ws_shifts, ws_nshifts, degens)
        for i in eachindex(h)
            ns = nshifts[i]
            
            frac = 1 / (ns * d)
            for is in 1:ns
                rcryst = R + shifts[i][is]
                h1 = out[rcryst]
                if h1 === nothing
                    h1 = TBBlock(rcryst, c * rcryst,
                                 zeros(ComplexF64, chk.n_wann, chk.n_wann),
                                 zeros(ComplexF64, chk.n_wann, chk.n_wann))
                    push!(out, h1)
                end
                h1.block[i] += h[i] * frac
                h1.tb_block[i] = h[i]
            end
            
        end
    end
    
    # this doesn't do much but it should make things more hermitian
    for o in out
        other = out[-o.R_cryst]
        other.block .= (o.block' .+ other.block)./2
        o.block .= (other.block' .+ o.block)./2
        other.tb_block .= (o.tb_block' .+ other.tb_block)./2
        o.tb_block .= (other.tb_block' .+ o.tb_block)./2
    end
    
    return out
end

for f in (:getindex, :size, :similar)
    @eval Base.$f(h::TBBlock, args...) = $f(block(h), args...)
end

LinearAlgebra.eigen(h::TBBlock) = eigen(block(h))


Base.getindex(h::TBHamiltonian, R::Vec3{Int}) = getfirst(x -> x.R_cryst == R, h)

#some small type piracy?
Base.zeros(m::AbstractArray{T}) where {T} = fill!(similar(m), zero(T))

zeros_block(h::TBHamiltonian) = zeros(block(h[1]))

similar_block(h::TBHamiltonian) = similar(block(h[1]))

blocksize(h::TBHamiltonian, args...) = size(block(h[1]), args...)

for op in (:+, :-, :*, :/)
    @eval function Base.$op(t::TBBlock{T}, v::T) where {T}
        return TBBlock(t.R_cart, t.R_cryst, $op(block(t), v), $op(t.tb_block, v))
    end
    @eval function Base.$op(v::T, t::TBBlock{T}) where {T}
        return TBBlock(t.R_cart, t.R_cryst, $op(v, block(t)), $op(v, t.tb_block))
    end
    @eval function Base.$op(t::TBBlock{T,M}, v::M) where {T,M}
        return TBBlock(t.R_cart, t.R_cryst, $op(block(t), v), $op(t.tb_block, v))
    end
    @eval function Base.$op(v::M, t::TBBlock{T,M}) where {T,M}
        return TBBlock(t.R_cart, t.R_cryst, $op(v, block(t)), $op(v, t.tb_block))
    end
    @eval function Base.$op(t::TBBlock{T,M}, v::TBBlock{T,M}) where {T,M}
        return TBBlock(t.R_cart, t.R_cryst, $op(block(t), block(v)),
                       $op(t.tb_block, v.tb_block))
    end
end

struct RmnBlock{T<:AbstractFloat}
    R_cart  ::Vec3{T}
    R_cryst ::Vec3{Int}
    block   ::Matrix{Point3{T}}
end

const TBRmn{T} = Vector{RmnBlock{T}}

