struct RmnBlock{T<:AbstractFloat}
    R_cart  ::Vec3{T}
    R_cryst ::Vec3{Int}
    block   ::Matrix{Point3{T}}
end

const TbRmn{T} = Vector{RmnBlock{T}}

struct TbBlock{T<:AbstractFloat,LT,M<:AbstractMatrix{Complex{T}}}
    R_cryst  :: Vec3{Int}
    R_cart   :: Vec3{LT}
    block    :: M #has contributions from all the ws shifts and divided by the degeneracies 1/(degen * nshifts), use for k-point interpolation
    tb_block :: M #this is the correct tight binding block, with ws shifts applied such that entries represent hopping between closest two wfs
end

const TbHami{T,LT,M} = Vector{TbBlock{T,LT,M}}
block(x::TbBlock) = x.block

for f in (:getindex, :size, :similar)
    @eval Base.$f(h::TbBlock, args...) = $f(block(h), args...)
end

LinearAlgebra.eigen(h::TbBlock) = eigen(block(h))


Base.getindex(h::TbHami, R::Vec3{Int}) = getfirst(x -> x.R_cryst == R, h)

#some small type piracy?
Base.zeros(m::AbstractArray{T}) where {T} = fill!(similar(m), zero(T))

zeros_block(h::TbHami) = zeros(block(h[1]))

similar_block(h::TbHami) = similar(block(h[1]))

blocksize(h::TbHami, args...) = size(block(h[1]), args...)

for op in (:+, :-, :*, :/)
    @eval function Base.$op(t::TbBlock{T}, v::T) where {T}
        return TbBlock(t.R_cart, t.R_cryst, $op(block(t), v), $op(t.tb_block, v))
    end
    @eval function Base.$op(v::T, t::TbBlock{T}) where {T}
        return TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)), $op(v, t.tb_block))
    end
    @eval function Base.$op(t::TbBlock{T,M}, v::M) where {T,M}
        return TbBlock(t.R_cart, t.R_cryst, $op(block(t), v), $op(t.tb_block, v))
    end
    @eval function Base.$op(v::M, t::TbBlock{T,M}) where {T,M}
        return TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)), $op(v, t.tb_block))
    end
    @eval function Base.$op(t::TbBlock{T,M}, v::TbBlock{T,M}) where {T,M}
        return TbBlock(t.R_cart, t.R_cryst, $op(block(t), block(v)),
                       $op(t.tb_block, v.tb_block))
    end
end

