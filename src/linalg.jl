struct ThreadCache{T}
    caches::Vector{T}
    ThreadCache(orig::T) where {T} =
        new{T}([deepcopy(orig) for i = 1:nthreads()])
end

@inline cache(t::ThreadCache) =
    t.caches[threadid()]

for f in (:getindex, :setindex!, :copyto!, :size, :length, :iterate, :sum, :view, :fill!)
    @eval Base.$f(t::ThreadCache{<:AbstractArray}, i...) = Base.$f(cache(t), i...)
end

for op in (:+, :-, :*, :/)
    @eval Base.$op(t::ThreadCache{T}, v::T) where {T} = $op(cache(t), v)
    @eval Base.$op(v::T, t::ThreadCache{T}) where {T} = $op(v, cache(t))
end

fillall!(t::ThreadCache{<:AbstractArray{T}}, v::T)   where {T} =
    fill!.(t.caches, (v,))

Base.reduce(op, t::ThreadCache; kwargs...) = reduce(op, t.caches; kwargs...)

LinearAlgebra.mul!(t1::ThreadCache{T}, v::T, t2::ThreadCache{T}) where {T<:AbstractArray} =
    mul!(cache(t1), v, cache(t2))

LinearAlgebra.mul!(t1::T, v::T, t2::ThreadCache{T}) where {T<:AbstractArray} =
    mul!(t1, v, cache(t2))

LinearAlgebra.mul!(t1::ThreadCache{T}, t2::ThreadCache{T}, t3::ThreadCache{T}) where {T<:AbstractArray} =
    mul!(cache(t1), cache(t2), cache(t3))

LinearAlgebra.adjoint!(t1::ThreadCache{T}, v::T) where {T} =
    adjoint!(cache(t1), v)

Base.ndims(::Type{ThreadCache{T}}) where {T<:AbstractArray} =
    ndims(T)
Base.Broadcast.broadcastable(tc::ThreadCache{<:AbstractArray}) =
    cache(tc)

@inline function eigen!(vecs::AbstractMatrix, ws::HermitianEigenWs)
    return Eigen(decompose!(ws, 'V', 'A', 'U', vecs, 0., 0., 0, 0, 1e-16)...)
end
@inline function eigen!(vals, vecs::AbstractMatrix, ws::HermitianEigenWs)
    ws.w = vals
    te = Eigen(decompose!(ws, 'V', 'A', 'U', vecs, 0., 0., 0, 0, 1e-16)...)
    return te
end

@inline function eigen!(vals::AbstractVector, vecs::ColinMatrix, c::HermitianEigenWs)
    n = div(length(vals),2)
    n2 = div(length(vecs),2)
    te = eigen!(up(vecs), c)
    copyto!(vals, te.values)
    copyto!(vecs, te.vectors)
    te = eigen!(down(vecs), c)
    copyto!(vals, n+1, te.values, 1, n)
    copyto!(vecs, n2+1, te.vectors, 1, n2)
    return Eigen(vals, vecs)
end

@inline function eigen!(vals::AbstractVector, vecs::NonColinMatrix, c::HermitianEigenWs)
    c.w = vals.data
    te = eigen!(vecs.data, c)
    return Eigen(vals, NonColinMatrix(te.vectors))
end

  
@inline function eigen(vecs::AbstractMatrix{T}, c::HermitianEigenWs{T}) where {T}
    return eigen!(copy(vecs), c)
end

@inline function eigen(vecs::AbstractMagneticMatrix{T}, c::HermitianEigenWs{T}) where {T}
    out  = copy(vecs)
    vals = MagneticVector(similar(out, T.parameters[1], size(out, 2)))
    return eigen!(vals, out, c)
end

@inline function eigen(h::AbstractMagneticMatrix)
    return eigen(h, HermitianEigenWs(h))
end

function Base.Matrix(e::Eigen{CT,T,<:ColinMatrix{CT}}) where {CT, T}
    d = size(e.vectors, 1)
    return ColinMatrix([e.vectors[1:d, 1:d] * diagm(0 => e.values[1:d]) * e.vectors[1:d, 1:d]' e.vectors[1:d, d + 1:2d] * diagm(0 => e.values[d + 1:2d]) * e.vectors[1:d, d + 1:2d]'])
end

Base.Array(e::Eigen{CT,T,<:ColinMatrix{CT}}) where {CT, T} = Matrix(e)
