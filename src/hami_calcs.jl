"Takes the hami in hopping param form and constructs H(k) from it"
function hami_from_k(hami_raw::Array{Tuple{Int,Int,Int,Int,Int,Complex{T}}},k_points::Array) where T<:AbstractFloat
  dim = 0
  for i=1:length(hami_raw)
    d = hami_raw[i][4]
    if d>dim
      dim = d
    else
      break
    end
  end
  out =zeros(Complex{T},(dim,dim))
  for i=1:size(hami_raw)[1]
    h = hami_raw[i]
    complex_part = 2 * pi * (k_points[1] * h[1] + k_points[2] * h[2] + k_points[3] * h[3])
    if h[4] == h[5]
        out[h[4], h[5]] += h[6] * cos(complex_part)
    else
        out[h[4], h[5]] += h[6] * exp(-1im * complex_part)
    end
  end

  return out
end


"Constructs the total spin-orbit-coupled Hamiltonian out of supplied angular momentums between the Wannier functions and uses the l_soc of the atoms."
function construct_soc_hami(hami, structure::WanStructure{T})::Matrix{Complex{T}} where T
    dim = getwandim(structure)
    Lx_soc = zeros(Complex{T}, dim, dim)
    Ly_soc = zeros(Complex{T}, dim, dim)
    Lz_soc = zeros(Complex{T}, dim, dim)
    i = 1
    for at in structure.atoms
        len = length(at.wfcs)-1
        Lx_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 1)
        Ly_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 2)
        Lz_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 3)
        i += len + 1
    end
    Lx_soc = (Lx_soc+Lx_soc')/2
    Ly_soc = (Ly_soc+Ly_soc')/2
    Lz_soc = (Lz_soc+Lz_soc')/2
    out = [hami+Lz_soc Lx_soc-1im*Ly_soc;Lx_soc+1im*Ly_soc hami-Lz_soc]
    return out
end

function heisenberg_energy(moments::Vector{<:Vec3}, exchanges::Vector{Matrix{T}}, H, anisotropy) where T
    energy = zero(T)
    nexch  = length(exchanges)
    nmom   = length(moments)
    for i = 1:nmom
        for j = i+1:nmom
            for k = 1:nexch
                energy += exchanges[k][i,j] * dot(moments[i], moments[j]) + H * moments[i][1] + anisotropy * moments[i][3]^2
            end
        end
    end
    return energy
end
