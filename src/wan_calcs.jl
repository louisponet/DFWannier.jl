"Calculates the angular momentum between two wavefunctions and around the center."
function calc_angmom(wfc1::WannierFunction{2, T}, wfc2::WannierFunction{2, T}, center::Point3{T}) where T <: AbstractFloat
	points = wfc1.points
    origin = points[1, 1, 1]
    da = points[2, 1, 1] - origin
    db = points[1, 2, 1] - origin
    dc = points[1, 1, 2] - origin
    V = SMatrix{3,3}(inv([convert(Array, da) convert(Array, db) convert(Array, dc)])')
    L = zero(Point3{Complex{T}})
    @inbounds for i2 = 2:size(wfc1)[3]
        @inbounds for i1 = 2:size(wfc1)[2]
            @inbounds for i = 2:size(wfc1)[1]
                dw_cryst = Point3(wfc2.values[i, i1, i2] - wfc2.values[i-1, i1, i2],
                                  wfc2.values[i, i1, i2] - wfc2.values[i, i1-1, i2],
  	  	  	  		              wfc2.values[i, i1, i2] - wfc2.values[i, i1, i2-1])

                r = points[i, i1, i2] - center

                dw_cart = V * dw_cryst
                L  += (wfc1.values[i, i1, i2]',) .* cross(r, dw_cart)
	        end
	    end
	end
	return -1im*L
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




"Calculates the angular momenta between all the supplied wavefunctions"
function calc_angmoms(wfcs::Array{Wfc3D{T}}) where T<:AbstractFloat
  out = Array{Array{Complex{T},1},2}((size(wfcs)[1],size(wfcs)[1]))
  for (i,wfc1) in enumerate(wfcs)
    for (i1,wfc2) in enumerate(wfcs)
      out[i,i1]=[calc_angmom(wfc1,wfc2)...]
    end
  end
  return [out fill([zero(Complex{T}) for i=1:3],size(out));fill([zero(Complex{T}) for i=1:3],size(out)) out]
end

function calc_angmoms!(structure::WanStructure{T}) where T
    for at in atoms(structure)
        if angmom(at)[1, 1] != zero(Point3{Complex{T}})
            continue
        end
        dim = length(wfcs(at))
        ang = Matrix{Vec3{Complex{T}}}(I, dim, dim)
        for i=1:dim, j=1:dim
            ang[i, j] = Vec3(calc_angmom(wfcs(at)[i], wfcs(at)[j], position(at))...)
        end
        setangmom!(at, ang)
    end
end

function calc_spins(structure::WanStructure{T}) where T<: AbstractFloat
    # if haskey(structure.data, :Sx)
    #     return structure.data[:Sx], structure.data[:Sy], structure.data[:Sz]
    # end
    dim = getwandim(structure)
    s_x = pauli(T, :x, 2*dim)
    s_y = pauli(T, :y, 2*dim)
    s_z = pauli(T, :z, 2*dim)
    data(structure)[:Sx] = s_x
    data(structure)[:Sy] = s_y
    data(structure)[:Sz] = s_z
    return s_x, s_y, s_z
end

"Calculates the spins between the supplied wavefunctions"
function calc_spins(wfcs::Array{<:Wfc3D{T},1}) where T<:AbstractFloat
  dim = length(wfcs)
  s_x = pauli(T,:x,2*dim)
  s_y = pauli(T,:y,2*dim)
  s_z = pauli(T,:z,2*dim)
  return s_x,s_y,s_z
end

"Calculates the dipole term between two wavefunctions. Make sure the wavefunctions are normalized!"
function calc_dip(wfc1::WannierFunction{N, T}, wfc2::WannierFunction{N, T}, points::Array{Point3{T}, 3}) where {N, T <: AbstractFloat}
  out = zero(Point3{Complex{T}})
  for i in eachindex(points)
    out += wfc1.values[i]' * wfc2.values[i] * points[i]
  end
  return real(out)
end

"Calculates the dipoles from the supplied wannier dipole output."
function calc_k_dips(dip_raw::Array{Tuple{Int,Int,Int,Int,Int,Point3{T}}}, k_points::AbstractArray) where T<:AbstractFloat
  dim = 0
  for i=1:length(dip_raw)
    d = dip_raw[i][4]
    if d>dim
      dim = d
    else
      break
    end
  end
  out = zeros(Point3{T},(dim,dim))
  tmp = [[zero(Complex{T}),zero(Complex{T}),zero(Complex{T})] for i=1:dim,i1=1:dim]
  for i=1:size(dip_raw)[1]
    d = dip_raw[i]
    complex_part = 2*pi*(k_points[1]*d[1]+k_points[2]*d[2]+k_points[3]*d[3])
    factor = exp(-1im*complex_part)
    tmp[d[4],d[5]][1] += d[6][1]*factor
    tmp[d[4],d[5]][2] += d[6][2]*factor
    tmp[d[4],d[5]][3] += d[6][3]*factor
  end
  for i in eachindex(out)
    out[i]=Point3(real(tmp[i][1]),real(tmp[i][2]),real(tmp[i][3]))
  end
  return Mat{2*dim, 2*dim, Point3{T}}([out zeros(out);zeros(out) out])
end
