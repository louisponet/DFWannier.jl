# function calculate_angmom_kernel(wfc1::CuDeviceArray{T,3},wfc2::CuDeviceArray{T,3},points::CuDeviceArray{Tuple{T,T,T},3},dim::CuDeviceArray{UInt32,1},center,V,out_x,out_y,out_z,n1,n2) where T
#   i,i1,i2 = threadID3D()

#   dim_a = dim[1]
#   dim_b = dim[2]
#   dim_c = dim[3]
#   if i > 1 && i1 > 1 && i2 > 1 && i < dim_a && i1 < dim_b && i2 < dim_c 
#     center_x = center[1]
#     center_y = center[2]
#     center_z = center[3]
#     dadx = V[1,1]
#     dbdx = V[2,1]
#     dcdx = V[3,1]
#     dady = V[1,2]
#     dbdy = V[2,2]
#     dcdy = V[3,2]
#     dadz = V[1,3]
#     dbdz = V[2,3]
#     dcdz = V[3,3]
#     wf1 = wfc1[i,i1,i2]
#     c_wfc1 = conj(wf1)
#     wf2 = wfc2[i,i1,i2]
#     diff_x = wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2]
#     diff_y = wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2]
#     diff_z = wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1]
#     point = points[i,i1,i2]
#     ddx = diff_x*dadx+diff_y*dbdx+diff_z*dcdx

#     ddy = diff_x*dady+diff_y*dbdy+diff_z*dcdy

#     ddz = diff_x*dadz+diff_y*dbdz+diff_z*dcdz

#     out_x[i,i1,i2] = c_wfc1*-1im*((point[2]-center_y)*ddz-(point[3]-center_z)*ddy)
#     out_y[i,i1,i2] = c_wfc1*-1im*((point[3]-center_z)*ddx-(point[1]-center_x)*ddz)
#     out_z[i,i1,i2] = c_wfc1*-1im*((point[1]-center_x)*ddy-(point[2]-center_y)*ddx)
#     n1[i,i1,i2]    = c_wfc1*wf1
#     n2[i,i1,i2]    = conj(wf2)*wf2
#   elseif i <= dim_a && i1 <= dim_b && i2 <= dim_c 
#     out_x[i,i1,i2] = zero(T) 
#     out_y[i,i1,i2] = zero(T)
#     out_z[i,i1,i2] = zero(T)
#     n1[i,i1,i2]    = zero(T)
#     n2[i,i1,i2]    = zero(T)
#   end
#   return nothing
# end

function calculate_angmom_kernel(wfc1::CuDeviceArray{T,3},wfc2::CuDeviceArray{T,3},points,dim,center,V,out_x,out_y,out_z,n1,n2) where T
  i,i1,i2 = threadID3D()
  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
  shmem_wfc = @cuDynamicSharedMem(T,(blockDim().x+2,blockDim().y+2,blockDim().z+2)) 
  if i > 1 && i1 > 1 && i2 > 1 && i < dim_a && i1 < dim_b && i2 < dim_c 
    if threadIdx().x==1 && threadIdx().y==1 && threadIdx().z==1
      bl_dx = blockDim().x
      bl_dy = blockDim().y
      bl_dz = blockDim().z
      
      # for m = 1:bl_dx
        # for m1 = 1:bl_dy
          # for m2=1:bl_dz
            shmem_wfc[UInt32(1),UInt32(1),UInt32(1)] = zero(T)
          # end
        # end
      # end
    end
    center_x = center[1]
    center_y = center[2]
    center_z = center[3]
    dadx = V[1,1]
    dbdx = V[2,1]
    dcdx = V[3,1]
    dady = V[1,2]
    dbdy = V[2,2]
    dcdy = V[3,2]
    dadz = V[1,3]
    dbdz = V[2,3]
    dcdz = V[3,3]
    wf1 = wfc1[i,i1,i2]
    wf2 = wfc2[i,i1,i2]
    c_wfc1 = conj(wf1)
    diff_x = wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2]
    diff_y = wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2]
    diff_z = wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1]
    
    point = points[i,i1,i2]
    out_x[i,i1,i2] = c_wfc1*-1im*((point[2]-center_y)*(diff_x*dadz+diff_y*dbdz+diff_z*dcdz)-(point[3]-center_z)*(diff_x*dady+diff_y*dbdy+diff_z*dcdy))
    out_y[i,i1,i2] = c_wfc1*-1im*((point[3]-center_z)*(diff_x*dadx+diff_y*dbdx+diff_z*dcdx)-(point[1]-center_x)*(diff_x*dadz+diff_y*dbdz+diff_z*dcdz))
    out_z[i,i1,i2] = c_wfc1*-1im*((point[1]-center_x)*(diff_x*dady+diff_y*dbdy+diff_z*dcdy)-(point[2]-center_y)*(diff_x*dadx+diff_y*dbdx+diff_z*dcdx))
    n1[i,i1,i2]    = c_wfc1*wf1
    n2[i,i1,i2]    = conj(wf2)*wf2
    
  elseif i <= dim_a && i1 <= dim_b && i2 <= dim_c 
    out_x[i,i1,i2] = zero(T) 
    out_y[i,i1,i2] = zero(T)
    out_z[i,i1,i2] = zero(T)
    n1[i,i1,i2]    = zero(T)
    n2[i,i1,i2]    = zero(T)
  end
  return nothing
end
"Calculates the angular momenta between two wavefunctions, around the atom of the second wavefunction."
function calculate_angmom(wfc1::Wfc3D_gpu{T},wfc2::Wfc3D_gpu{T},V,center::CuArray{T,1},dims,Lx::CuArray{Complex{T},3},Ly::CuArray{Complex{T},3},Lz::CuArray{Complex{T},3},n1::CuArray{Complex{T},3},n2::CuArray{Complex{T},3}) where T<:AbstractFloat
  blocks,threads = get_blocks_threads(wfc1.values)
  @cuda (blocks,threads,96000) calculate_angmom_kernel(wfc1.values,wfc2.values,wfc2.grid,dims,center,V,Lx,Ly,Lz,n1,n2)
  
  n=sqrt(sum(n1)*sum(n2))
  return (sum(Lx)/n,sum(Ly)/n,sum(Lz)/n)
end

function calculate_angmom(wfc1::Wfc3D_gpu{T},wfc2::Wfc3D_gpu{T}) where T<:AbstractFloat
  if wfc1.atom.center != wfc2.atom.center
    return zeros(Complex{T},3)
  end
  dims = size(wfc1.values)
  Lx = CuArray(zeros(Complex{T},dims))
  Ly = CuArray(zeros(Complex{T},dims))
  Lz = CuArray(zeros(Complex{T},dims))
  n1 = CuArray(zeros(Complex{T},dims))
  n2 = CuArray(zeros(Complex{T},dims))
  
  grid = Array(wfc1.grid)
  origin = grid[1,1,1]
  a = grid[2,1,1] .- origin
  b = grid[1,2,1] .- origin
  c = grid[1,1,2] .- origin
  V = CuArray(inv([[a...] [b...] [c...]]))
  blocks,threads = get_blocks_threads(wfc1.values)
  dim = CuArray(UInt32[size(wfc1.grid)...])
  cu_center     = CuArray([wfc1.atom.center.x,wfc1.atom.center.y,wfc1.atom.center.z])
  dim = CuArray(UInt32[size(wfc1.grid)...])
  @cuda (blocks,threads) calculate_angmom_kernel(wfc1.values,wfc2.values,wfc2.grid,dim,cu_center,V,Lx,Ly,Lz,n1,n2)
  n= sqrt(sum(n1)*sum(n2))
  ls = [sum(Lx),sum(Ly),sum(Lz)]
  return ls./n 
end

function calculate_angmoms(wfcs::Array{Wfc3D_gpu{T}},V,centers,dims,Lx,Ly,Lz,n1,n2) where T<:AbstractFloat
  out = Array{Array{Complex{T},1},2}((length(wfcs),length(wfcs)))
  for (i,wfc1) in enumerate(wfcs)
    center1 = wfc1.atom.center
    for (i1,wfc2) in enumerate(wfcs)
      center2 = wfc2.atom.center
      if center1 == center2
        out[i,i1]=[calculate_angmom(wfc1,wfc2,V,centers[i1],dims,Lx,Ly,Lz,n1,n2)...]
      else
        out[i,i1]=[zero(Complex{T}) for i =1:3]
      end
    end
  end
  return [out fill([zero(Complex{T}) for i=1:3],size(out));fill([zero(Complex{T}) for i=1:3],size(out)) out]
end