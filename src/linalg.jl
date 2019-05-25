import LinearAlgebra.LAPACK: syev!, @blasfunc, BlasInt, chkstride1, checksquare, chklapackerror, liblapack


# We use Upper Triangular blas for everything! And Eigvals are always all calculated
function blas_eig_ccall(A     ::AbstractMatrix{ComplexF32},
	                    W     ::Vector{Float32},
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
function blas_eig_ccall(A     ::AbstractMatrix{ComplexF64},
	                    W     ::Vector{Float64},
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

function LinearAlgebra.eigen!(vals::Vector{T}, vecs::AbstractMatrix{Complex{T}}, c::EigCache{T}) where {T}
	blas_eig_ccall(vecs, vals, c.work, c.lwork, c.rwork, c.n, c.info)
	return Eigen(vals, vecs)
end

function LinearAlgebra.eigen(vecs::AbstractMatrix{Complex{T}}, c::EigCache{T}) where {T}
	out = copy(vecs)
	vals = similar(out, T, size(out, 1))
	blas_eig_ccall(out, vals, c.work, c.lwork, c.rwork, c.n, c.info)
	return Eigen(vals, out)
end
