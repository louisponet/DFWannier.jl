import LinearAlgebra.LAPACK: syev!, @blasfunc, BlasInt, chkstride1, checksquare, chklapackerror, liblapack
import LinearAlgebra.BLAS: libblas
import LinearAlgebra: eigen, eigen!
import Base: @propagate_inbounds

abstract type Spin end
struct Up <: Spin end
struct Down <: Spin end

"Represents a magnetic Hamiltonian matrix with the block structure [up updown;downup down]"
abstract type AbstractMagneticMatrix{T} <: AbstractMatrix{T} end

data(m::AbstractMatrix) = m
data(m::AbstractMagneticMatrix) = m.data

Base.similar(::Type{M}, i::NTuple{2,Int}) where {M <: AbstractMagneticMatrix} =
    M(Matrix{M.parameters[1]}(undef, i))

for f in (:length, :size, :setindex!, :elsize)
    @eval @inline @propagate_inbounds Base.$f(c::AbstractMagneticMatrix, args...) =
    Base.$f(c.data, args...)
end

Base.pointer(c::AbstractMagneticMatrix, i::Integer) = pointer(c.data, i)

"Magnetic block dimensions"
blockdim(c::AbstractMatrix) = div(size(data(c), 2), 2)

up(c::AbstractMatrix) =   (d = blockdim(c); view(data(c), 1:d, 1:d))
down(c::AbstractMatrix) = (d = blockdim(c); r = d + 1:2 * d; view(data(c), r, r))

# Standard getindex behavior

for f in (:view, :getindex)
    @eval @inline @propagate_inbounds Base.$f(c::AbstractMagneticMatrix, args...) =
        getindex(c.data, args...)
    @eval @inline @propagate_inbounds Base.$f(c::AbstractMagneticMatrix, r::Union{Colon, AbstractUnitRange}, i::Int) =
        MagneticVector(Base.$f(c.data, r, i))
    @eval @inline @propagate_inbounds Base.$f(c::AbstractMagneticMatrix, i::Int, r::Union{Colon, AbstractUnitRange}) =
        MagneticVector(Base.$f(c.data, i, r))
end

Base.similar(c::M, args::AbstractUnitRange...) where {M <: AbstractMagneticMatrix} =
    M(similar(c.data), args...)

"""
    ColinMatrix{T, M <: AbstractMatrix{T}} <: AbstractMagneticMatrix{T}

Defines a Hamiltonian Matrix with [up zeros; zeros down] structure.
It is internally only storing the up and down block.
"""
struct ColinMatrix{T,M <: AbstractMatrix{T}} <: AbstractMagneticMatrix{T}
    data::M
end

function ColinMatrix(up::AbstractMatrix, down::AbstractMatrix)
    @assert size(up) == size(down)
    return ColinMatrix([up down])
end

Base.Array(c::ColinMatrix{T}) where T =
    (d = blockdim(c); [c[Up()] zeros(T, d, d); zeros(T, d, d) c[Down()]])

down(c::ColinMatrix) = (d = blockdim(c); view(c.data, 1:d, d + 1:2 * d))
blockdim(c::ColinMatrix) = size(c.data, 1)

function LinearAlgebra.diag(c::ColinMatrix)
    d = blockdim(c)
    r = LinearAlgebra.diagind(d, d)
    [c[r];c[r.+last(r)]]
end

"""
    NonColinMatrix{T, M <: AbstractMatrix{T}} <: AbstractMagneticMatrix{T}


Defines a Hamiltonian Matrix with [up updown;downup down] structure.
Since atomic projections w.r.t spins are defined rather awkwardly in Wannier90 for exchange calculations,
i.e. storing the up-down parts of an atom sequentially,
a NonColinMatrix reshuffles the entries of a matrix such that it follows the above structure. 
"""
struct NonColinMatrix{T,M <: AbstractMatrix{T}} <: AbstractMagneticMatrix{T}
    data::M
end

"Reshuffles standard Wannier90 up-down indices to the ones for the structure of a NonColinMatrix."
function Base.convert(::Type{NonColinMatrix}, m::M) where {M <: AbstractMatrix}
    @assert iseven(size(m, 1)) "Error, dimension of the supplied matrix is odd, i.e. it does not contain both spin components."
    data = similar(m)
    d    = blockdim(m)
    for i in 1:2:size(m, 1), j in 1:2:size(m, 2) 
        up_id1 = div1(i, 2) 
        up_id2 = div1(j, 2) 
        data[up_id1, up_id2] = m[i, j] 
        data[up_id1 + d, up_id2] = m[i + 1, j] 
        data[up_id1, up_id2 + d] = m[i, j + 1] 
        data[up_id1 + d, up_id2 + d] = m[i + 1, j + 1]
    end
    return NonColinMatrix(data)
end

function NonColinMatrix(up::AbstractMatrix{T}, down::AbstractMatrix{T}) where {T}
    @assert size(up) == size(down)
    return NonColinMatrix([up zeros(T, size(up));zeros(T, size(up)) down])
end

Base.Array(c::NonColinMatrix) = copy(c.data)
   
function uprange(a::DFC.Projection)
    projrange = range(a)
    if length(projrange) > a.orb.size 
        return range(div1(first(projrange), 2), length = div(length(projrange), 2))
    else
        return projrange
    end
end
uprange(a::DFC.AbstractAtom) = vcat(uprange.(projections(a))...)
    
## Indexing ##
for f in (:view, :getindex)
    @eval function Base.$f(c::ColinMatrix, a1::T, a2::T) where {T <: Union{DFC.Projection,DFC.AbstractAtom}}
        projrange1 = range(a1)
        projrange2 = range(a2)

        return ColinMatrix($f(c, projrange1, projrange2), $f(c, projrange1, projrange2 .+ blockdim(c)))
    end
    @eval function Base.$f(c::NonColinMatrix, a1::T, a2::T) where {T <: Union{DFC.Projection,DFC.AbstractAtom}}
        up_range1 = uprange(a1)
        up_range2 = uprange(a2)
        d = blockdim(c)
        dn_range1 = up_range1 .+ d
        dn_range2 = up_range2 .+ d
        return NonColinMatrix([$f(c, up_range1, up_range2) $f(c, up_range1, dn_range2)
                               $f(c, dn_range1, up_range2) $f(c, dn_range1, dn_range2)])
    end

    @eval Base.$f(c::AbstractMagneticMatrix, a1::T) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, a1, a1)

    @eval Base.$f(c::ColinMatrix, a1::T, a2::T, ::Up) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, range(a1), range(a2))
    
    @eval Base.$f(c::NonColinMatrix, a1::T, a2::T, ::Up) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, uprange(a1), uprange(a2))

    @eval Base.$f(c::ColinMatrix, a1::T, a2::T, ::Down) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, range(a1), range(a2) .+ blockdim(c))
        
    @eval Base.$f(c::NonColinMatrix, a1::T, a2::T, ::Down) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, uprange(a1) .+ blockdim(c), uprange(a2) .+ blockdim(c))
        
    @eval Base.$f(c::NonColinMatrix, a1::T, a2::T, ::Up, ::Down) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, uprange(a1), uprange(a2) .+ blockdim(c))

    @eval Base.$f(c::NonColinMatrix, a1::T, a2::T, ::Down, ::Up) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, uprange(a1) .+ blockdim(c), uprange(a2))
        
    @eval Base.$f(c::NonColinMatrix, ::Up, ::Down) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        (s = size(c,1); $f(c, 1:div(s, 2), div(s, 2)+1:s))

    @eval Base.$f(c::NonColinMatrix, ::Down, ::Up) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        (s = size(c,1); $f(c, div(s, 2)+1:s, 1:div(s, 2)))
        
    @eval Base.$f(c::AbstractMatrix, a1::T, a2::T, ::Up) where {T<:Union{DFC.Projection, DFC.AbstractAtom}} =
        $f(c, range(a1), range(a2))

    @eval Base.$f(c::AbstractMatrix, a1::T, a2::T, ::Down) where {T<:Union{DFC.Projection, DFC.AbstractAtom}} =
        $f(c, range(a1), range(a2) .+ blockdim(c))

    @eval Base.$f(c::AbstractMagneticMatrix, ::Up) =
        (r = 1:blockdim(c); $f(c, r, r))

    @eval Base.$f(c::ColinMatrix, ::Down) =
        (d = blockdim(c); r = 1:d; $f(c, r, r .+ d))
        
    @eval Base.$f(c::NonColinMatrix, ::Down) =
        (d = blockdim(c); r = d+1 : 2*d; $f(c, r, r))

    @eval Base.$f(c::AbstractMatrix, ::Up) =
        (r = 1:blockdim(c); $f(c, r, r))

    @eval Base.$f(c::AbstractMatrix, ::Down) =
        (d = blockdim(c); r = d + 1:2 * d; $f(c, r, r))
    
end

for op in (:*, :-, :+, :/)
    @eval @inline Base.$op(c1::ColinMatrix, c2::ColinMatrix) =
        ColinMatrix($op(c1[Up()], c2[Up()]), $op(c1[Down()], c2[Down()]))
    @eval @inline Base.$op(c1::NonColinMatrix, c2::NonColinMatrix) =
        NonColinMatrix($op(c1.data, c2.data))
end

    # BROADCASTING
Base.BroadcastStyle(::Type{<:AbstractMagneticMatrix}) =
    Broadcast.ArrayStyle{AbstractMagneticMatrix}()

Base.ndims(::Type{<:AbstractMagneticMatrix}) =
    2

Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{AbstractMagneticMatrix}}, ::Type{ElType}) where {ElType} =
    Base.similar(AbstractMagneticMatrix{ElType}, axes(bc))

Base.axes(c::AbstractMagneticMatrix) =
    Base.axes(c.data)

@inline @propagate_inbounds Base.broadcastable(c::AbstractMagneticMatrix) =
    c

@inline @propagate_inbounds Base.unsafe_convert(::Type{Ptr{T}}, c::AbstractMagneticMatrix{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, c.data)

for (elty, cfunc) in zip((:ComplexF32, :ComplexF64), (:cgemm_, :zgemm_))
    @eval @inline function LinearAlgebra.mul!(C::ColinMatrix{$elty}, A::ColinMatrix{$elty}, B::ColinMatrix{$elty})
        dim = blockdim(C)
        dim2 = dim * dim
        ccall((@blasfunc($(cfunc)), libblas), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                         Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                         Ref{BlasInt}),
                         'N', 'N', dim, dim,
                         dim, one($elty), A, dim,
                         B, dim, zero($elty), C, dim)
        ccall((@blasfunc($(cfunc)), libblas), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                         Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                         Ref{BlasInt}),
                         'N', 'N', dim, dim,
                         dim, one($elty), pointer(A, dim2 + 1), dim,
                         pointer(B, dim2 + 1), dim, zero($elty), pointer(C, dim2 + 1), dim)

        return C
    end
end

@inline function LinearAlgebra.adjoint(c::AbstractMagneticMatrix)
    out = similar(c)
    adjoint!(out, c)
end

@inline @inbounds function LinearAlgebra.adjoint!(out::ColinMatrix, in1::ColinMatrix)
    dim = blockdim(out)
    for i in 1:dim, j in 1:dim
        out[j, i] = in1[i, j]'
        out[j, i + dim] = in1[i, j + dim]'
    end
    return out
end

@inline LinearAlgebra.adjoint!(out::NonColinMatrix, in1::NonColinMatrix) =
    adjoint!(out.data, in1.data)

for (elty, relty, cfunc) in zip((:ComplexF32, :ComplexF64), (:Float32, :Float64), (:cheev_, :zheev_))
# We use Upper Triangular blas for everything! And Eigvals are always all calculated
    @eval @inline function blas_eig_ccall(A::AbstractMatrix{$elty},
                                          W::AbstractVector{$relty},
                                          work::Vector{$elty},
                                          lwork::BlasInt,
                                          rwork::Vector{$relty},
                                          n::Int,
                                          info::Ref{BlasInt})

    ccall((@blasfunc($cfunc), liblapack), Cvoid,
           (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
           Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt}),
           'V', 'U', n, A, n, W, work, lwork, rwork, info)
    chklapackerror(info[])
    end
end

"Vector following the same convention as the in AbstractMagneticMatrix, i.e. first half of the indices contain the up part, second the down part"
struct MagneticVector{T} <: AbstractVector{T}
    data::Vector{T}
end

for f in (:length, :size, :setindex!, :elsize)
    @eval @inline @propagate_inbounds Base.$f(c::MagneticVector, args...) =
    Base.$f(c.data, args...)
end

up(c::MagneticVector) = view(c.data, 1:div(length(c), 2))
down(c::MagneticVector) = (lc = length(c); view(c.data, div(lc, 2):lc))

# Standard getindex behavior
@inline @propagate_inbounds Base.getindex(c::MagneticVector, args...) =
    getindex(c.data, args...)

for f in (:view, :getindex)
    @eval @inline @propagate_inbounds Base.$f(c::MagneticVector, args::AbstractUnitRange...) =
    Base.$f(c.data, args...)
end

Base.similar(v::MagneticVector) = MagneticVector(similar(v.data))

"Reshuffles standard Wannier90 up-down indices to the ones for the structure of a MagneticVector."
function Base.convert(::Type{MagneticVector}, v::V) where {V <: AbstractVector}
    @assert iseven(length(v)) "Error, dimension of the supplied matrix is odd, i.e. it does not contain both spin components."
    data = similar(v)
    vl = length(v)
    d    =div(vl, 2)
    
    for i in 1:2:vl
        up_id1 = div1(i, 2) 
        data[up_id1] = v[i] 
        data[up_id1 + d] = v[i + 1] 
    end
    return NonColinMatrix(data)
end

## Indexing with atoms and spins ##
for f in (:view, :getindex)
    @eval function Base.$f(c::MagneticVector, a1::T) where {T <: Union{DFC.Projection,DFC.AbstractAtom}}
        projrange1 = uprange(a1)
        return MagneticVector([$f(c, projrange1); $f(c, projrange1 .+ div(length(c), 2))])
    end
    @eval Base.$f(c::MagneticVector, a1::T, ::Up) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, uprange(a1))
    
    @eval Base.$f(c::MagneticVector, a1::T, ::Down) where {T <: Union{DFC.Projection,DFC.AbstractAtom}} =
        $f(c, range(a1), uprange(a2) + div(length(c), 2))
        
    @eval Base.$f(c::MagneticVector, ::Up) =
        $f(c, 1:div(length(c), 2))

    @eval Base.$f(c::MagneticVector, ::Down) =
        (lc = length(c); $f(c, div(lc, 2) + 1:lc))
end

for op in (:*, :-, :+, :/)
    @eval @inline Base.$op(c1::MagneticVector, c2::MagneticVector) =
        MagneticVector($op(c1.data, c2.data))
end

struct EigCache{T <: AbstractFloat}
    work::Vector{Complex{T}}
    lwork::BlasInt
    rwork::Vector{T}
    n::Int
    info::Ref{BlasInt}
    function EigCache(A::AbstractMatrix{Complex{relty}}) where {relty}
        chkstride1(A)
        elty = Complex{relty}
        n = checksquare(A)
        W     = similar(A, relty, n)
        work  = Vector{elty}(undef, 1)
        lwork = BlasInt(-1)
        rwork = Vector{relty}(undef, max(1, 3n - 2))
        info  = Ref{BlasInt}()

        blas_eig_ccall(A, W, work, lwork, rwork, n, info)
        lwork = BlasInt(real(work[1]))
        resize!(work, lwork)
        return new{relty}(work, lwork, rwork, n, info)
    end
end
EigCache(A::ColinMatrix) = EigCache(up(A))
EigCache(A::NonColinMatrix) = EigCache(A.data)

@inline function eigen!(vals::AbstractVector{T}, vecs::AbstractMatrix{Complex{T}}, c::EigCache{T}) where {T}
    blas_eig_ccall(vecs, vals, c.work, c.lwork, c.rwork, c.n, c.info)
    return Eigen(vals, vecs)
end

@inline function eigen!(vals::AbstractVector{T}, vecs::ColinMatrix{Complex{T}}, c::EigCache{T}) where {T}
    eigen!(view(vals.data, 1:c.n), up(vecs), c)
    eigen!(view(vals.data, c.n + 1:2 * c.n), down(vecs), c)
    return Eigen(vals, vecs)
end

@inline function eigen!(vals::AbstractVector{T}, vecs::NonColinMatrix{Complex{T}}, c::EigCache{T}) where {T}
    eigen!(vals.data, vecs.data, c)
    return Eigen(vals, vecs)
end

    
@inline function eigen(vecs::AbstractMatrix{Complex{T}}, c::EigCache{T}) where {T}
    out  = copy(vecs)
    vals = similar(out, T, size(out, 2))
    return eigen!(vals, out, c)
end

@inline function eigen(vecs::AbstractMagneticMatrix{Complex{T}}, c::EigCache{T}) where {T}
    out  = copy(vecs)
    vals = MagneticVector(similar(out, T, size(out, 2)))
    return eigen!(vals, out, c)
end

# @inline function eigen!(orig::M) where {M<:AbstractMagneticMatrix}
#     vals, vecs = eigen!(orig.data)
#     return Eigen(vals, M(vecs)) 
# end

@inline function eigen(h::AbstractMagneticMatrix)
    return eigen(h, EigCache(h))
end
    # Very unoptimized
function Base.Array(e::Eigen{Complex{T},T,<:ColinMatrix{Complex{T}}}) where {T}
    d = size(e.vectors, 1)
    return [e.vectors[1:d, 1:d] * diagm(0 => e.values[1:d]) * e.vectors[1:d, 1:d]' e.vectors[1:d, d + 1:2d] * diagm(0 => e.values[d + 1:2d]) * e.vectors[1:d, d + 1:2d]']
end
