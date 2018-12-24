function find_start(wfc::Wfc3D_gpu,R,partitions)::Tuple{Tuple{Int64,Int64,Int64},Tuple{Int64,Int64,Int64}}
  part_1D = partitions^(1/3)
  # part_1D = partitions
  grid = Array(wfc.grid)
  dim_a = size(grid)[1]
  dim_b = size(grid)[2]
  dim_c = size(grid)[3]
  stride_a::Int64 = dim_a/part_1D
  stride_b::Int64 = dim_b/part_1D
  stride_c::Int64 = dim_c/part_1D
  anchors = [[grid[a,b,c]...] for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
  shifted_anchors = [[grid[a,b,c]...]-Array(R) for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
  for i in eachindex(anchors)
    for j in eachindex(shifted_anchors)
      if norm(anchors[i]-shifted_anchors[j])<0.01
        tmp1 = ind2sub(anchors,i)
        tmp2 = ind2sub(shifted_anchors,j)
        ind1 = ((tmp1[1]-1)*stride_a+1,(tmp1[2]-1)*stride_b+1,(tmp1[3]-1)*stride_c+1)
        ind2 = ((tmp2[1]-1)*stride_a+1,(tmp2[2]-1)*stride_b+1,(tmp2[3]-1)*stride_c+1)
        return ind1,ind2
      end
    end
  end
end


function find_start(grid::Array{<:Point3,3},R,partitions)::Tuple{Tuple{Int64,Int64,Int64},Tuple{Int64,Int64,Int64}}
  part_1D = partitions^(1/3)
  # part_1D = partitions
  dim_a = size(grid)[1]
  dim_b = size(grid)[2]
  dim_c = size(grid)[3]
  stride_a::Int64 = dim_a/part_1D
  stride_b::Int64 = dim_b/part_1D
  stride_c::Int64 = dim_c/part_1D
  anchors = [grid[a,b,c] for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
  shifted_anchors = [grid[a,b,c]-R for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
  for i in eachindex(anchors)
    for j in eachindex(shifted_anchors)
      if norm(anchors[i]-shifted_anchors[j])<0.00001
        tmp1 = ind2sub(anchors,i)
        tmp2 = ind2sub(shifted_anchors,j)
        ind1 = ((tmp1[1]-1)*stride_a+1,(tmp1[2]-1)*stride_b+1,(tmp1[3]-1)*stride_c+1)
        ind2 = ((tmp2[1]-1)*stride_a+1,(tmp2[2]-1)*stride_b+1,(tmp2[3]-1)*stride_c+1)
        return ind1,ind2
      end
    end
  end
end

function calc_inds_coeffs(wfc::Wfc3D_gpu{T},k) where T
  grid = [Point3(g...) for g in Array(wfc.grid)]
  indices = Array{Tuple{Tuple{Int32,Int32,Int32},Tuple{Int32,Int32,Int32}},1}()
  coefficients = Array{Complex{T},1}()
  for R1=-1:1,R2=-1:1,R3=-1:1
    if R1==0 && R2==0 && R3==0
      continue
    end
    R= R1*wfc.cell[1]+R2*wfc.cell[2]+R3*wfc.cell[3]
    ind1,ind2 = find_start(grid,R,27)
    push!(coefficients,Complex{T}(exp(dot(-2*pi*k,[R1,R2,R3])*1im)))
    push!(indices,((Int32(ind1[1]),Int32(ind1[2]),Int32(ind1[3])),(Int32(ind1[1]-ind2[1]),Int32(ind1[2]-ind2[2]),Int32(ind1[3]-ind2[3]))))
  end
  return indices,coefficients
end

function blocks_threads(arr::CuArray)
  total_threads = min(length(arr),attribute(CuDevice(0),MAX_THREADS_PER_BLOCK))
  max_threads_per_dim = floor(Int, total_threads^(1/ndims(arr)))
  blocks  = Int.(ceil.(size(arr)./max_threads_per_dim))
  threads = Int.(ceil.(size(arr)./blocks))
  return blocks,threads
end

@inline function threadID3D()::Tuple{UInt32,UInt32,UInt32}
  i = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
  i1 = (blockIdx().y-UInt32(1)) * blockDim().y + threadIdx().y
  i2 = (blockIdx().z-UInt32(1)) * blockDim().z + threadIdx().z
  return i,i1,i2
end

function bloch_kernel(wfc_orig::CuDeviceArray{Complex{T}, 3}, indices,coefficients,dim, out) where T
  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
  i1 = (blockIdx().x-1) * blockDim().x + threadIdx().x
  i2 = (blockIdx().y-1) * blockDim().y + threadIdx().y
  i3 = (blockIdx().z-1) * blockDim().z + threadIdx().z
  for (c,(ind1,ind2)) in zip(coefficients,indices)
    j1 = i1-ind2[1]
    j2 = i2-ind2[2]
    j3 = i3-ind2[3]
    if i1>= ind1[1] && j1 <= dim_a
      if i2>= ind1[2] && j2 <= dim_b
        if i3>= ind1[3] && j3 <= dim_c
          out[i1,i2,i3] += c*wfc_orig[j1,j2,j3]
        end
      end
    end
  end
  return nothing
end

function construct_bloch_sum(wfc::Wfc3D_gpu{T}, k::Array{T}) where T
  indices,coefficients = calc_inds_coeffs(wfc,k)
  out = Wfc3D_gpu(wfc.grid,copy(wfc.values),wfc.cell,wfc.atom)
  blocks,threads = blocks_threads(out.values)
  out = Wfc3D_gpu(wfc.grid,copy(wfc.values),wfc.cell,wfc.atom)
  @cuda (blocks,threads) bloch_kernel(wfc.values,CuArray(indices),CuArray(coefficients),CuArray(Int64[size(wfc.values)...]),out.values)
  return out
end

function construct_bloch_sums(orig_wfcs::Array{Wfc3D_gpu{T},1},k_wfcs::Array{Wfc3D_gpu{T},1}, k::Array{T},indices,coefficients) where T <: AbstractFloat
  blocks,threads = blocks_threads(orig_wfcs[1].values)
  dims   = CuArray(UInt32[size(orig_wfcs[1].values)...])
  for (n,wfc) in enumerate(orig_wfcs)
    k_wfcs[n].values .= wfc.values
    @cuda (blocks,threads) bloch_kernel(wfc.values,indices,coefficients,dims,k_wfcs[n].values)
  end
end

function calculate_angmom_kernel(wfc1::CuDeviceArray{Complex{T},3}, wfc2::CuDeviceArray{Complex{T},3}, points::CuDeviceArray{Tuple{T,T,T},3}, dim, center, V, out_x, out_y,out_z,n1,n2) where T
  i,i1,i2 = threadID3D()

  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
  if i > 1 && i1 > 1 && i2 > 1 && i < dim_a && i1 < dim_b && i2 < dim_c
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
    c_wfc1 = conj(wf1)
    wf2 = wfc2[i,i1,i2]
    diff_x = wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2]
    diff_y = wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2]
    diff_z = wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1]

    point = points[i,i1,i2]
    ddx = diff_x*dadx+diff_y*dbdx+diff_z*dcdx

    ddy = diff_x*dady+diff_y*dbdy+diff_z*dcdy

    ddz = diff_x*dadz+diff_y*dbdz+diff_z*dcdz

    out_x[i,i1,i2] = c_wfc1*-1im*((point[2]-center_y)*ddz-(point[3]-center_z)*ddy)
    out_y[i,i1,i2] = c_wfc1*-1im*((point[3]-center_z)*ddx-(point[1]-center_x)*ddz)
    out_z[i,i1,i2] = c_wfc1*-1im*((point[1]-center_x)*ddy-(point[2]-center_y)*ddx)
    n1[i,i1,i2]    = c_wfc1*wf1
    n2[i,i1,i2]    = conj(wf2)*wf2
  elseif i <= dim_a && i1 <= dim_b && i2 <= dim_c
    out_x[i,i1,i2] = zero(Complex{T})
    out_y[i,i1,i2] = zero(Complex{T})
    out_z[i,i1,i2] = zero(Complex{T})
    n1[i,i1,i2]    = zero(Complex{T})
    n2[i,i1,i2]    = zero(Complex{T})
  end
  return nothing
end

# function calculate_angmom_kernel(wfc1::CuDeviceArray{T,1},wfc2::CuDeviceArray{T,1},points::CuDeviceArray{Tuple{T,T,T},3},dim::CuDeviceArray{UInt32,1},center,V,out_x,out_y,out_z,n1,n2) where T
#   i = (blockIdx().x -1)*blockDim().x + threadIdx().x
#   i1 = i/(dim[2]*dim[3])
#   t =


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
"Calculates the angular momenta between two wavefunctions, around the atom of the second wavefunction."
function calculate_angmom(wfc1::Wfc3D_gpu{T},wfc2::Wfc3D_gpu{T},V,center::CuArray{T,1},dims,Lx::CuArray{Complex{T},3},Ly::CuArray{Complex{T},3},Lz::CuArray{Complex{T},3},n1::CuArray{Complex{T},3},n2::CuArray{Complex{T},3}) where T<:AbstractFloat
  blocks,threads = blocks_threads(wfc1.values)
  @cuda (blocks,threads) calculate_angmom_kernel(wfc1.values,wfc2.values,wfc2.grid,dims,center,V,Lx,Ly,Lz,n1,n2)

  n=sqrt(sum(n1)*sum(n2))
  return (sum(Lx)/n,sum(Ly)/n,sum(Lz)/n)
end

function calculate_angmom(wfc1::Wfc3D_gpu{T},wfc2::Wfc3D_gpu{T}) where T<:AbstractFloat
  if wfc1.atom.center != wfc2.atom.center
    return zeros(Complex{T},3)
  end
  dims = size(wfc1.values)
  Lx = CuArray(zeros(Complex{T}, dims))
  Ly = CuArray(zeros(Complex{T}, dims))
  Lz = CuArray(zeros(Complex{T}, dims))
  n1 = CuArray(zeros(Complex{T}, dims))
  n2 = CuArray(zeros(Complex{T}, dims))

  grid = Array(wfc1.grid)
  origin = grid[1,1,1]
  a = grid[2,1,1] .- origin
  b = grid[1,2,1] .- origin
  c = grid[1,1,2] .- origin
  V = CuArray(inv([[a...] [b...] [c...]]))
  blocks,threads = blocks_threads(wfc1.values)
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










#this is all trash




# function find_start(wfc::Wfc3D_gpu,R,partitions)::Tuple{Int64,Int64,Int64}
#   part_1D = partitions^(1/3)
#   # part_1D = partitions
#   grid = Array(wfc.grid)
#   dim_a = size(grid)[1]
#   dim_b = size(grid)[2]
#   dim_c = size(grid)[3]
#   stride_a::Int64 = dim_a/part_1D
#   stride_b::Int64 = dim_b/part_1D
#   stride_c::Int64 = dim_c/part_1D
#   # anchors = [[grid[a,b,c]...] for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
#   #hack
#   shifted_anchors_neg = [[grid[a,b,c]...]+Array(R) for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
#   shifted_anchors_pos = [[grid[a,b,c]...]-Array(R) for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
#   # for i in eachindex(anchors)
#   for j in eachindex(shifted_anchors_pos)
#     #hack
#     if norm([grid[1]...]-shifted_anchors_neg[j])<0.01
#       tmp2 = ind2sub(shifted_anchors_neg,j)
#       ind2 = ((tmp2[1]-1)*stride_a+1,(tmp2[2]-1)*stride_b+1,(tmp2[3]-1)*stride_c+1)
#       return ind2
#     elseif  norm([grid[1]...]-shifted_anchors_pos[j])<0.01
#       tmp2 = ind2sub(shifted_anchors_pos,j)
#       ind2 = ((tmp2[1]-1)*stride_a+1,(tmp2[2]-1)*stride_b+1,(tmp2[3]-1)*stride_c+1)
#       return ind2
#     end
#   end
# end

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

# function calculate_angmom_kernel(wfc1::CuDeviceArray{T,3},wfc2::CuDeviceArray{T,3},points,dim,center,V,out_x,out_y,out_z,n1,n2) where T
#   i,i1,i2 = threadID3D()


#   dim_a = dim[1]
#   dim_b = dim[2]
#   dim_c = dim[3]
#   shmem_wfc2 = @cuDynamicSharedMem(T,(blockDim().x+2,blockDim().y+2,blockDim().z+2))
#   if i > 1 && i1 > 1 && i2 > 1 && i < dim_a && i1 < dim_b && i2 < dim_c
#     if threadIdx().x==1 && threadIdx().y==1 &&threadIdx().z==1
#       bl_dx = blockDim().x
#       bl_dy = blockDim().y
#       bl_dz = blockDim().z
#       for m = -1:bl_dx-1
#       for m1 = -1:bl_dy-1
#       for m2=-1:bl_dz-1
#         shmem_wfc2[m+2,m1+2,m2+2] = wfc2[i+m,i1+m2,i2+m2]
#       end
#       end
#       end
#     end
#     sync_threads()
#     center_x = center[1]
#     center_y = center[2]
#     center_z = center[3]
#     ind_x =threadIdx().x+1
#     ind_y =threadIdx().y+1
#     ind_z =threadIdx().z+1
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
#     wf2 = shmem_wfc2[ind_x,ind_y,ind_z]
#     c_wfc1 = conj(wf1)
#     # diff_x = wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2]
#     # diff_y = wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2]
#     # diff_z = wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1]
#     diff_x = shmem_wfc2[ind_x+1,ind_y,ind_z]-shmem_wfc2[ind_x-1,ind_y,ind_z]
#     diff_y = shmem_wfc2[ind_x,ind_y+1,ind_z]-shmem_wfc2[ind_x,ind_y-1,ind_z]
#     diff_z = shmem_wfc2[ind_x,ind_y,ind_z+1]-shmem_wfc2[ind_x,ind_y,ind_z-1]

#     point = points[i,i1,i2]
#     out_x[i,i1,i2] = c_wfc1*-1im*((point[2]-center_y)*(diff_x*dadz+diff_y*dbdz+diff_z*dcdz)-(point[3]-center_z)*(diff_x*dady+diff_y*dbdy+diff_z*dcdy))
#     out_y[i,i1,i2] = c_wfc1*-1im*((point[3]-center_z)*(diff_x*dadx+diff_y*dbdx+diff_z*dcdx)-(point[1]-center_x)*(diff_x*dadz+diff_y*dbdz+diff_z*dcdz))
#     out_z[i,i1,i2] = c_wfc1*-1im*((point[1]-center_x)*(diff_x*dady+diff_y*dbdy+diff_z*dcdy)-(point[2]-center_y)*(diff_x*dadx+diff_y*dbdx+diff_z*dcdx))
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

# function calculate_angmom(wfc1::Wfc3D_gpu{T},wfc2::Wfc3D_gpu{T},V,center::CuArray{T,1},dims,Lx::CuArray{Complex{T},3},Ly::CuArray{Complex{T},3},Lz::CuArray{Complex{T},3},n1::CuArray{Complex{T},3},n2::CuArray{Complex{T},3}) where T<:AbstractFloat
#   blocks,threads = blocks_threads(wfc1.values)
#   @cuda (blocks,threads,prod(threads .+ 2)*sizeof(Complex{T})) calculate_angmom_kernel(wfc1.values,wfc2.values,wfc2.grid,dims,center,V,Lx,Ly,Lz,n1,n2)
#   n=sqrt(sum(n1)*sum(n2))
#   return (sum(Lx)/n,sum(Ly)/n,sum(Lz)/n)
# end

# function calculate_angmom(wfc1::Wfc3D_gpu{T},wfc2::Wfc3D_gpu{T}) where T<:AbstractFloat
#   if wfc1.atom.center != wfc2.atom.center
#     return zeros(Complex{T},3)
#   end
#   dims = size(wfc1.values)
#   Lx = CuArray(zeros(Complex{T},dims))
#   Ly = CuArray(zeros(Complex{T},dims))
#   Lz = CuArray(zeros(Complex{T},dims))
#   n1 = CuArray(zeros(Complex{T},dims))
#   n2 = CuArray(zeros(Complex{T},dims))

#   grid = Array(wfc1.grid)
#   origin = grid[1,1,1]
#   a = grid[2,1,1] .- origin
#   b = grid[1,2,1] .- origin
#   c = grid[1,1,2] .- origin
#   V = CuArray(inv([[a...] [b...] [c...]]))
#   blocks,threads = blocks_threads(wfc1.values)
#   dim = CuArray(UInt32[size(wfc1.grid)...])
#   cu_center     = CuArray([wfc1.atom.center.x,wfc1.atom.center.y,wfc1.atom.center.z])
#   dim = CuArray(UInt32[size(wfc1.grid)...])
#   @cuda (blocks,threads) calculate_angmom_kernel(wfc1.values,wfc2.values,wfc2.grid,dim,cu_center,V,Lx,Ly,Lz,n1,n2)
#   n= sqrt(sum(n1)*sum(n2))
#   ls = [sum(Lx),sum(Ly),sum(Lz)]
#   return ls./n
# end

# function calculate_angmoms(wfcs::Array{Wfc3D_gpu{T}},V,centers,dims,Lx,Ly,Lz,n1,n2) where T<:AbstractFloat
#   out = Array{Array{Complex{T},1},2}((length(wfcs),length(wfcs)))
#   for (i,wfc1) in enumerate(wfcs)
#     center1 = wfc1.atom.center
#     for (i1,wfc2) in enumerate(wfcs)
#       center2 = wfc2.atom.center
#       if center1 == center2
#         out[i,i1]=[calculate_angmom(wfc1,wfc2,V,centers[i1],dims,Lx,Ly,Lz,n1,n2)...]
#       else
#         out[i,i1]=[zero(Complex{T}) for i =1:3]
#       end
#     end
#   end
#   return [out fill([zero(Complex{T}) for i=1:3],size(out));fill([zero(Complex{T}) for i=1:3],size(out)) out]
# end

# function bloch_kernel(wfc_orig::CuDeviceArray{Complex{T},3}, indices,coefficients, out) where T
#   # dim_a = dim[1]
#   # dim_b = dim[2]
#   # dim_c = dim[3]
#   # i1 = (blockIdx().x-1) * blockDim().x*per_thread[1] + threadIdx().x
#   # i2 = (blockIdx().y-1) * blockDim().y*per_thread[2] + threadIdx().y
#   # i3 = (blockIdx().z-1) * blockDim().z*per_thread[3] + threadIdx().z

#   dim_a =81
#   dim_b =81
#   dim_c =81
#   i1 = (blockIdx().x-1) * blockDim().x*2 + threadIdx().x
#   i2 = (blockIdx().y-1) * blockDim().y*2 + threadIdx().y
#   i3 = (blockIdx().z-1) * blockDim().z*2 + threadIdx().z
#   for (c,(ind1,ind2)) in zip(coefficients,indices)
#     j1 = i1-ind2[1]
#     j2 = i2-ind2[2]
#     j3 = i3-ind2[3]
#     # if i1>= ind1[1] && j1+per_thread[1] <= dim_a
#     #   if i2>= ind1[2] && j2+per_thread[2] <= dim_b
#     #     if i3>= ind1[3] && j3+per_thread[3]<= dim_c
#     #       for n1=0:per_thread[1]
#     #         for n2=0:per_thread[2]
#     #           for n3=0:per_thread[3]
# if i1>= ind1[1] && j1+2 <= dim_a
#       if i2>= ind1[2] && j2+2 <= dim_b
#         if i3>= ind1[3] && j3+2<= dim_c
#           for n1=0:2
#             for n2=0:2
#               for n3=0:2
#                 out[i1+n1,i2+n2,i3+n3] += c*wfc_orig[j1+n1,j2+n2,j3+n3]
#               end
#             end
#           end
#         end
#       end
#     end
#   end
#   return nothing
# end

# function construct_bloch_sums(orig_wfcs::Array{Wfc3D_gpu{T},1},k_wfcs::Array{Wfc3D_gpu{T},1}, k::Array{T},indices,coefficients) where T <: AbstractFloat
#   blocks,threads = blocks_threads(orig_wfcs[1].values)
#   # dims   = CuArray(UInt32[size(orig_wfcs[1].values)...])
#   # per_thread = CuArray(UInt32[2,2,2])
#   for (n,wfc) in enumerate(orig_wfcs)
#     k_wfcs[n].values .= wfc.values
#     # @cuda ((11,11,11),(4,4,4)) bloch_kernel(wfc.values,indices,coefficients,dims,per_thread,k_wfcs[n].values)
#     @cuda ((11,11,11),(4,4,4)) bloch_kernel(wfc.values,indices,coefficients,k_wfcs[n].values)
#   end
# end
