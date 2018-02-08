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
  out = MMatrix{dim, dim, Complex{T}}(zeros(Complex{T},(dim,dim)))
  for i=1:size(hami_raw)[1]
    h = hami_raw[i]
    complex_part = 2*pi*(k_points[1]*h[1]+k_points[2]*h[2]+k_points[3]*h[3])
    if h[4] == h[5]
        out[h[4],h[5]] += real(h[6]*exp(-1im*complex_part))
    else
        out[h[4],h[5]] += h[6]*exp(-1im*complex_part)
    end
  end

  return SMatrix{dim, dim, Complex{T}}(Hermitian(out))
end

"Constructs the total spin-orbit-coupled Hamiltonian out of supplied angular momentums between the Wannier functions and uses the l_soc of the atoms."
function construct_soc_hami(hami,angmoms,atoms)
  soc_angmoms = [0.5*atoms[i].l_soc*angmoms[i,j] for i=1:length(atoms),j=1:length(atoms)]
  h_Lx,h_Ly,h_Lz = map(x->x[1],soc_angmoms),map(x->x[2],soc_angmoms),map(x->x[3],soc_angmoms)
  h_Lx = (h_Lx+h_Lx')/2
  h_Ly = (h_Ly+h_Ly')/2
  h_Lz = (h_Lz+h_Lz')/2
  out = [hami+h_Lz h_Lx-1im*h_Ly;h_Lx+1im*h_Ly hami-h_Lz]
  return out
end

"Constructs the total spin-orbit-coupled Hamiltonian out of supplied angular momentums between the Wannier functions and uses the l_soc of the atoms."
function construct_soc_hami(hami, structure::WanStructure{T}) where T
    dim = getwandim(structure)
    Lx_soc = MMatrix{dim, dim, Complex{T}}()
    Ly_soc = similar(Lx_soc)
    Lz_soc = similar(Lx_soc)
    i = 1
    for at in structure.atoms
        len = length(at.wfcs)-1
        Lx_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 1)
        Ly_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 2)
        Lz_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 3)
        i += len
    end
    Lx_soc = (Lx_soc+Lx_soc')/2
    Ly_soc = (Ly_soc+Ly_soc')/2
    Lz_soc = (Lz_soc+Lz_soc')/2
    out = [hami+Lz_soc Lx_soc-1im*Ly_soc;Lx_soc+1im*Ly_soc hami-Lz_soc]
    return out
end
#Used by bloch calculations
function construct_L_hamis{T<:AbstractFloat}(wfcs::Array{Wfc3D{T},1})
  dim = length(wfcs)
  hami_Lx = zeros(Complex{T},(dim,dim))
  hami_Ly = zeros(Complex{T},(dim,dim))
  hami_Lz = zeros(Complex{T},(dim,dim))
  for i=1:dim
    for i1=1:dim
      Lx1,Ly1,Lz1 = calculate_angmom(wfcs[i],wfcs[i1])
      hami_Lx[i,i1] = wfcs[i1].atom.l_soc * Lx1
      hami_Ly[i,i1] = wfcs[i1].atom.l_soc * Ly1
      hami_Lz[i,i1] = wfcs[i1].atom.l_soc * Lz1
    end
  end
  hami_Lx = (hami_Lx+hami_Lx')/2
  hami_Ly = (hami_Ly+hami_Ly')/2
  hami_Lz = (hami_Lz+hami_Lz')/2
  return hami_Lx,hami_Ly,hami_Lz
end

function construct_soc_hami(hami,wfcs)
  h_Lx,h_Ly,h_Lz = construct_L_hamis(wfcs)
  out = [hami+0.5h_Lz 0.5h_Lx-0.5im*h_Ly;0.5h_Lx+0.5im*h_Ly hami-0.5h_Lz]
  return out
end
#Not sure if used
"Constructs the angular momentum part of the Tight Binding Hamiltonians"
function construct_L_hamis(angmoms,l1,l2)
  dim = div(size(angmoms[1])[1],2)
  dim_2 = div(dim,2)
  hami_Lx = zeros(Complex{T},(dim,dim))
  hami_Ly = zeros(Complex{T},(dim,dim))
  hami_Lz = zeros(Complex{T},(dim,dim))
  for i1 = 1:dim
    for i = 1:dim
      hami_Lx[i,i1] = angmoms[1][i,i1][1]*l1+angmoms[2][i,i1][1]*l2
      hami_Ly[i,i1] = angmoms[1][i,i1][2]*l1+angmoms[2][i,i1][2]*l2
      hami_Lz[i,i1] = angmoms[1][i,i1][3]*l1+angmoms[2][i,i1][3]*l2
    end
  end
  hami_Lx = (hami_Lx+hami_Lx')/2
  hami_Ly = (hami_Ly+hami_Ly')/2
  hami_Lz = (hami_Lz+hami_Lz')/2
  return hami_Lx,hami_Ly,hami_Lz
end
