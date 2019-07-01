import LinearAlgebra.LAPACK: syev!, @blasfunc, BlasInt, chkstride1, checksquare, chklapackerror, liblapack
import LinearAlgebra.BLAS: libblas
import LinearAlgebra: eigen, eigen!
import Base: @propagate_inbounds

abstract type Spin end
struct Up <: Spin end
struct Down <: Spin end 

struct ColinMatrix{T, M <: AbstractArray{T, 2}} <: AbstractMatrix{T}
	data::M
end

function ColinMatrix(up::AbstractMatrix, down::AbstractMatrix)
	@assert size(up) == size(down)
	ColinMatrix([up down])
end

Base.similar(::Type{DFWannier.ColinMatrix{Complex{Float64},M} where M<:AbstractArray{Complex{Float64},2}}, i::Tuple{Int64,Int64}) =
	ColinMatrix(Matrix{ComplexF64}(undef, i))

up(c::ColinMatrix)   = view(c.data, 1:blockdim(c), 1:blockdim(c))
# down(c::ColinMatrix) = view(c.data, blockdim(c)+1:2*blockdim(c), 1:blockdim(c))
down(c::ColinMatrix) = view(c.data, 1:blockdim(c), blockdim(c)+1:2*blockdim(c))
# blockdim(c::ColinMatrix) = size(c.data, 2)
blockdim(c::ColinMatrix) = size(c.data, 1)

for f in (:length, :size, :setindex!, :elsize)
	@eval @inline @propagate_inbounds Base.$f(c::ColinMatrix, args...) = Base.$f(c.data, args...)
end

@inline @propagate_inbounds Base.getindex(c::ColinMatrix, args...) = getindex(c.data, args...)

for f in (:view, :getindex)
	@eval @inline @propagate_inbounds Base.$f(c::ColinMatrix{T, M} where {T, M<:AbstractMatrix{T}}, args::AbstractUnitRange...) =
		Base.$f(c.data, args...)
end

Base.similar(c::ColinMatrix{T, M} where {T, M}, args::AbstractUnitRange...) =
	ColinMatrix(similar(c.data), args...)

for f in (:view, :getindex)
	@eval function Base.$f(c::ColinMatrix, a1::T, a2::T) where {T<:Union{DFC.Projection, DFC.AbstractAtom}}
		projrange1 = range(a1)
		projrange2 = range(a2)

		return ColinMatrix($f(c, projrange1, projrange2), $f(c, projrange1, projrange2 .+ blockdim(c)))
	end

	@eval Base.$f(c::ColinMatrix, a1::T, a2::T, ::Up) where {T<:Union{DFC.Projection, DFC.AbstractAtom}} =
		$f(c, range(a1), range(a2))

	@eval Base.$f(c::ColinMatrix, a1::T, a2::T, ::Down) where {T<:Union{DFC.Projection, DFC.AbstractAtom}} =
		$f(c, range(a1), range(a2) .+ blockdim(c))

	@eval Base.$f(c::ColinMatrix, ::Up) =
		$f(c, 1:blockdim(c), 1:blockdim(c))

	@eval Base.$f(c::ColinMatrix, ::Down) =
		$f(c, 1:blockdim(c), (1:blockdim(c)) .+ blockdim(c))
end

Base.BroadcastStyle(::Type{<:ColinMatrix}) =
	Broadcast.ArrayStyle{ColinMatrix}()

Base.ndims(::Type{<:ColinMatrix}) =
	2
	
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ColinMatrix}}, ::Type{ElType}) where {ElType} =
	Base.similar(ColinMatrix{ElType}, axes(bc))
	

Base.axes(c::ColinMatrix) =
	Base.axes(c.data)

@inline @propagate_inbounds Base.broadcastable(c::ColinMatrix)      = c

@inline @propagate_inbounds Base.unsafe_convert(::Type{Ptr{T}}, c::ColinMatrix{T}) where {T} = Base.unsafe_convert(Ptr{T}, c.data)

@inline function LinearAlgebra.mul!(C::ColinMatrix{ComplexF32}, A::ColinMatrix{ComplexF32}, B::ColinMatrix{ComplexF32})
	dim = blockdim(C)
	dim2 = dim*dim
	ccall((@blasfunc(cgemm_), libblas), Cvoid,
	                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
	                 Ref{BlasInt}, Ref{ComplexF32}, Ptr{ComplexF32}, Ref{BlasInt},
	                 Ptr{ComplexF32}, Ref{BlasInt}, Ref{ComplexF32}, Ptr{ComplexF32},
	                 Ref{BlasInt}),
	                 'N', 'N', dim, dim,
	                 dim, one(ComplexF32), A, dim,
	                 B, dim, zero(ComplexF32), C, dim)
	ccall((@blasfunc(cgemm_), libblas), Cvoid,
	                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
	                 Ref{BlasInt}, Ref{ComplexF32}, Ptr{ComplexF32}, Ref{BlasInt},
	                 Ptr{ComplexF32}, Ref{BlasInt}, Ref{ComplexF32}, Ptr{ComplexF32},
	                 Ref{BlasInt}),
	                 'N', 'N', dim, dim,
	                 dim, one(ComplexF32), pointer(A, dim2+1), dim,
	                 pointer(B, dim2+1), dim, zero(ComplexF32), pointer(C,dim2+1), dim)

    C
end

@inline function LinearAlgebra.mul!(C::ColinMatrix{ComplexF64}, A::ColinMatrix{ComplexF64}, B::ColinMatrix{ComplexF64})
	dim = blockdim(C)
	dim2 = dim*dim

	ccall((@blasfunc(zgemm_), libblas), Cvoid,
	                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
	                 Ref{BlasInt}, Ref{ComplexF64}, Ptr{ComplexF64}, Ref{BlasInt},
	                 Ptr{ComplexF64}, Ref{BlasInt}, Ref{ComplexF64}, Ptr{ComplexF64},
	                 Ref{BlasInt}),
	                 'N', 'N', dim, dim,
	                 dim, one(ComplexF64), A, dim,
	                 B, dim, zero(ComplexF64), C, dim)

	ccall((@blasfunc(zgemm_), libblas), Cvoid,
	                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
	                 Ref{BlasInt}, Ref{ComplexF64}, Ptr{ComplexF64}, Ref{BlasInt},
	                 Ptr{ComplexF64}, Ref{BlasInt}, Ref{ComplexF64}, Ptr{ComplexF64},
	                 Ref{BlasInt}),
	                 'N', 'N', dim, dim,
	                 dim, one(ComplexF64), pointer(A, dim2+1), dim,
	                 pointer(B, dim2+1), dim, zero(ComplexF64), pointer(C,dim2+1), dim)

    C
end

@inline function LinearAlgebra.adjoint!(out::ColinMatrix, in1::ColinMatrix)
	dim = blockdim(out)
	@inbounds begin
		for i = 1:dim, j = 1:dim
			out[j, i] = in1[i, j]'
			out[j, i+dim] = in1[i, j+dim]'
		end
	end
	out
end

# We use Upper Triangular blas for everything! And Eigvals are always all calculated
@inline function blas_eig_ccall(A     ::AbstractMatrix{ComplexF32},
	                    W     ::AbstractVector{Float32},
                        work  ::Vector{ComplexF32},
                        lwork ::BlasInt,
                        rwork ::Vector{Float32},
                        n     ::Int,
                        info  ::Ref{BlasInt})

	ccall((@blasfunc(cheev_), liblapack), Cvoid,
	          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{ComplexF32}, Ref{BlasInt},
	          Ptr{Float32}, Ptr{ComplexF32}, Ref{BlasInt}, Ptr{Float32}, Ptr{BlasInt}),
	          'V', 'U', n, A, n, W, work, lwork, rwork, info)
	chklapackerror(info[])
end
@inline function blas_eig_ccall(A     ::AbstractMatrix{ComplexF64},
	                    W     ::AbstractVector{Float64},
                        work  ::Vector{ComplexF64},
                        lwork ::BlasInt,
                        rwork ::Vector{Float64},
                        n     ::Int,
                        info  ::Ref{BlasInt})

	ccall((@blasfunc(zheev_), liblapack), Cvoid,
	          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{ComplexF64}, Ref{BlasInt},
	          Ptr{Float64}, Ptr{ComplexF64}, Ref{BlasInt}, Ptr{Float64}, Ptr{BlasInt}),
	          'V', 'U', n, A, n, W, work, lwork, rwork, info)
	chklapackerror(info[])
end

struct EigCache{T <: AbstractFloat}
	work   ::Vector{Complex{T}}
	lwork  ::BlasInt
	rwork  ::Vector{T}
	n      ::Int
	info   ::Ref{BlasInt}
	function EigCache(A::AbstractMatrix{Complex{relty}}) where {relty}
		chkstride1(A)
		elty = Complex{relty}
	    n = checksquare(A)
	    W     = similar(A, relty, n)
	    work  = Vector{elty}(undef, 1)
	    lwork = BlasInt(-1)
	    rwork = Vector{relty}(undef, max(1, 3n-2))
	    info  = Ref{BlasInt}()

		blas_eig_ccall(A, W, work, lwork, rwork, n, info)
	    lwork = BlasInt(real(work[1]))
	    resize!(work, lwork)
	    return new{relty}(work, lwork, rwork, n, info)
    end
end
EigCache(A::ColinMatrix) = EigCache(up(A))

@inline function eigen!(vals::AbstractVector{T}, vecs::AbstractMatrix{Complex{T}}, c::EigCache{T}) where {T}
	blas_eig_ccall(vecs, vals, c.work, c.lwork, c.rwork, c.n, c.info)
	return Eigen(vals, vecs)
end

@inline function eigen!(vals::AbstractVector{T}, vecs::ColinMatrix{Complex{T}}, c::EigCache{T}) where {T}
    eigen!(view(vals, 1:c.n), up(vecs), c)
    eigen!(view(vals, c.n+1:2*c.n), down(vecs), c)
	return Eigen(vals, vecs)
end


@inline function eigen(vecs::AbstractMatrix{Complex{T}}, c::EigCache{T}) where {T}
	out  = copy(vecs)
	vals = similar(out, T, size(out, 2))
	return eigen!(vals, out, c)
end

#Very unoptimized
function Base.Array(e::Eigen{Complex{T}, T, <:ColinMatrix{Complex{T}}}) where {T}
	d = size(e.vectors, 1)
	return [e.vectors[1:d, 1:d] * diagm(0=>e.values[1:d]) * e.vectors[1:d, 1:d]' e.vectors[1:d, d+1:2d] * diagm(0=>e.values[d+1:2d]) * e.vectors[1:d, d+1:2d]']
end


