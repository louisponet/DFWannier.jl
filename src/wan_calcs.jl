"Calculates the angular momentum between two wavefunctions and around the center."
function calc_angmom(wfc1::WannierFunction{N, T}, wfc2::WannierFunction{N, T}, center::Point3{T}) where {N, T <: AbstractFloat}
	points = wfc1.points
    origin = points[1, 1, 1]
    da     = points[2, 1, 1] - origin
    db     = points[1, 2, 1] - origin
    dc     = points[1, 1, 2] - origin
    V      = SMatrix{3,3}(inv([convert(Array, da) convert(Array, db) convert(Array, dc)])')
    L      = zero(Point3{Complex{T}})

    @inbounds for i2 = 2:size(wfc1)[3]
        for i1 = 2:size(wfc1)[2]
            for i = 2:size(wfc1)[1]
                dw_cryst = Point3(wfc2.values[i, i1, i2] - wfc2.values[i-1, i1,   i2],
                                  wfc2.values[i, i1, i2] - wfc2.values[i,   i1-1, i2],
  	  	  	  		              wfc2.values[i, i1, i2] - wfc2.values[i,   i1,   i2-1])

                r       = points[i, i1, i2] - center
                dw_cart = V * dw_cryst
                L      += (wfc1.values[i, i1, i2]',) .* cross(r, dw_cart)
	        end
	    end
	end
	return -1im * L
end

function calc_spin(wfc1::WannierFunction{2, T}, wfc2::WannierFunction{2, T}) where T <: AbstractFloat
	S = Point3(SMatrix{2, 2}(0, 1, 1, 0)/2, 
	           SMatrix{2, 2}(0, -1im, 1im, 0)/2,
	           SMatrix{2, 2}(1, 0, 0, -1)/2)

	outS = zero(Point3{Complex{T}})
	for (w1, w2) in zip(wfc1.values, wfc2.values)
		outS += (w1',) .* S .* (w2,)
	end
	return outS
end


"Calculates the dipole term between two wavefunctions. Make sure the wavefunctions are normalized!"
function calc_dip(wfc1::WannierFunction{N, T}, wfc2::WannierFunction{N, T}) where {N, T <: AbstractFloat}
	out = zero(Point3{Complex{T}})
	for (w1, w2, p) in zip(wfc1, wfc2, wfc1.points)
		out += w1' * w2 * p
	end
	return real(out)
end

# Is this code actually correct?
# "Calculates the dipoles from the supplied wannier dipole output."
# function calc_k_dips(dip_raw::Array{Tuple{Int, Int, Int, Int, Int, Point3{T}}}, k_points::AbstractArray) where T<:AbstractFloat
# 	dim = 0
# 	for i=1:length(dip_raw)
# 		d = dip_raw[i][4]
# 		if d > dim
# 			dim = d
# 		else
# 			break
# 		end
# 	end
# 	out = zeros(Point3{T}, dim, dim)
# 	tmp = [[zeros(Complex{T}, 3)] for i=1:dim, i1=1:dim]
# 	for i=1:size(dip_raw)[1]
# 		d = dip_raw[i]
# 		complex_part = 2π*(k_points[1]*d[1]+k_points[2]*d[2]+k_points[3]*d[3])
# 		factor = exp(-1im * complex_part)
# 		tmp[d[4],d[5]][1] += d[6][1] * factor
# 		tmp[d[4],d[5]][2] += d[6][2] * factor
# 		tmp[d[4],d[5]][3] += d[6][3] * factor
# 	end
# 	for i in eachindex(out)
# 		out[i] = Point3(real(tmp[i][1]),real(tmp[i][2]),real(tmp[i][3]))
# 	end
# 	return Mat{2*dim, 2*dim, Point3{T}}([out zeros(out);zeros(out) out])
# end
