import Base: getindex, zero, show, -, +, ==, !=, *, /, view

struct WannierFunction{N, T<:AbstractFloat} <: AbstractArray{SVector{N, Complex{T}}, 3}
	points::Array{Point{3, T}, 3}
	values::Array{SVector{N, Complex{T}}, 3}
end

function WannierFunction(filename::AbstractString, points::Array{Point3{T}, 3}=read_points_from_xsf(filename)) where {T <: AbstractFloat}
	re = read_values_from_xsf(T, filename)
	values = [SVector(complex(a)) for a in re]
	return normalize(WannierFunction(points, values))
end

function WannierFunction(filename_re::String, filename_im::String, points::Array{Point3{T}, 3} = read_points_from_xsf(filename_re)) where {T <: AbstractFloat}
	re, im = read_values_from_xsf.(T, (filename_re, filename_im))
	values = [SVector(Complex(a, b)) for (a, b) in zip(re, im)]
	return normalize(WannierFunction(points, values))
end

function WannierFunction(filename_up_re::String, filename_up_im::String, filename_down_re::String, filename_down_im::String, points::Array{Point3{T}, 3} = read_points_from_xsf(filename_up_re)) where {T <: AbstractFloat}
	up_re, up_im, down_re, down_im =
		read_values_from_xsf.(T, (filename_up_re, filename_up_im, filename_down_re, filename_down_im))
	values = [SVector(Complex(a, b), Complex(c, d)) for (a, b, c, d) in zip(up_re, up_im, down_re, down_im)]
	return normalize(WannierFunction(points, values))
end

WannierFunction(point_func::Function,  points::Array) =
    normalize(WannierFunction(points, point_func.(points)))

for f in (:size, :getindex, :setindex!)
	@eval @inline @propagate_inbounds Base.$f(x::WannierFunction, i...) =
		Base.$f(x.values, i...)
end

for f in (:length, :stride, :ndims, :axes, :strides)
	@eval @inline Base.$f(w::WannierFunction) = Base.$f(w.values)
end

Base.similar(x::WannierFunction,::Type{S}) where S = 
  WannierFunction(x.points, similar(x.values, S))

Base.unsafe_convert(T::Type{<:Ptr}, x::WannierFunction) =
	unsafe_convert(T, x.values)


Base.Broadcast.broadcastable(w::WannierFunction) =
	w.values

#### LinearAlgebra overloads
function LinearAlgebra.adjoint(w::WannierFunction)
	out = WannierFunction(w.points, similar(w.values))
	adjoint!(out, w)
end

LinearAlgebra.adjoint!(w1::WannierFunction, w2::WannierFunction) =
	w1 .= adjoint.(w2)

function LinearAlgebra.dot(w1::WannierFunction{T}, w2::WannierFunction{T}) where {T}
    s = zero(T)
    for (v1, v2) in zip(w1.values, w2.values)
        s += v1' * v2
    end
    return real(s)
end

function LinearAlgebra.dot(v::Vector, wfs::Vector{<:WannierFunction})
    res = WannierFunction(wfs[1].points, zeros(eltype(wfs[1].values), size(wfs[1].values)))
    for ic in 1:length(v)
        res .+= v[ic] .* wfs[ic]
    end
    return res
end
LinearAlgebra.dot(wfs::Vector{<:WannierFunction}, v::Vector) =
    dot(v, wfs)

LinearAlgebra.norm(wfc::WannierFunction) =
	dot(wfc, wfc)

LinearAlgebra.normalize!(wfc::WannierFunction) =
	wfc ./= sqrt(norm(wfc))

same_grid(w1::WannierFunction, w2::WannierFunction) =
	w1.points === w2.points

function wan_op(op::Function, w1::W, w2::W) where {W <: WannierFunction}
	@assert same_grid(w1, w2) "Wannier functions are not defined on the same grid"
	return WannierFunction(w1.points, op(w1.values, w2.values))
end

+(w1::WannierFunction, w2::WannierFunction) = wan_op(+, w1, w2)
*(w1::WannierFunction, w2::WannierFunction) = wan_op(*, w1, w2)
-(w1::WannierFunction, w2::WannierFunction) = wan_op(-, w1, w2)

*(w1::WannierFunction, n::Number) = WannierFunction(w1.points, w1.values .* n)
*(n::Number, w1::WannierFunction) = WannierFunction(w1.points, n .* w1.values)
/(w1::WannierFunction, n::Number) = WannierFunction(w1.points, n ./ w1.values)
/(n::Number, w1::WannierFunction) = WannierFunction(w1.points, w1.values ./ n)

LinearAlgebra.dot(w1::WannierFunction, n::Number) = w1 * n
LinearAlgebra.dot(n::Number, w1::WannierFunction) = n * w1

function bloch_sum(wfunc, kpoint)
    cell_boundaries = div.(size(wfunc.points), 3)
    bloch = WannierFunction(wfunc.points, copy(wfunc.values))
    dims = size(wfunc.values)
    for i1 in -1:1:1, i2 in -1:1:1, i3 in -1:1:1
        R_cryst = Vec3(i1, i2, i3)
        o1, o2, o3 = cell_boundaries .* R_cryst
        phase = ℯ^(2im*π*(R_cryst ⋅ kpoint)) 
        if i1 + i2 + i3 == 0
            continue
        end
        for j3 in 1:dims[3]
            oid3 = j3 - o3
            if !(0 < oid3 <= dims[3])
                continue
            end
            for j2 in 1:dims[2]
                oid2 = j2 - o2
                if !(0 < oid2 <= dims[2])
                    continue
                end
                for j1 in 1:dims[1]
                    oid1 = j1 - o1
                    if !(0 < oid1 <= dims[1])
                        continue
                    end
                    bloch.values[j1, j2, j3] += phase * wfunc.values[oid1, oid2, oid3]
                end
            end
        end
    end
    return bloch
end
