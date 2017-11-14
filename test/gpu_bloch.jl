using DFWannier
T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...],true);
x2 = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);

@time bands = calculate_eig_angmom_soc_bloch_gpu(x,x.k_points[1:10]);
@time test = calculate_eig_angmom_soc_bloch(x2,x2.k_points[1:10]);
test= calculate_angmom(x.wfcs[3],x.wfcs[3])
test1= calculate_angmom(x2.wfcs[3],x2.wfcs[3])


using CUDAdrv,CUDAnative
CUDAdrv.version()
dev = CuDevice(0)

ctx = CuContext(dev)
function Lx(wfc1,wfc2,point1,point2,center,V,Lx)
  dady = V[1,2]
  dbdy = V[2,2]
  dcdy = V[3,2]
  dadz = V[1,3]
  dbdz = V[2,3]
  dcdz = V[3,3]
  dday = (wfc2[1])*dady
  ddby = (wfc2[2])*dbdy
  ddcy = (wfc2[3])*dcdy
  ddy = dday+ddby+ddcy
  
  ddaz = (wfc2[1])*dadz
  ddbz = (wfc2[2])*dbdz
  ddcz = (wfc2[3])*dcdz
  ddz = ddaz+ddbz+ddcz
  return CUDAnative.conj(wfc1)*-1im*((point1[2]-center[2])*ddz-(point1[3]-center[3])*ddy)
end

function angmom_grid(wfc1::CuDeviceArry{Complex{T}},wfc2::CuDeviceArry{Complex{T}},points,dim,center,V,out,n1,n2) where {T}
  Lx = zero(Complex{T})
  Ly = zero(Complex{T})
  Lz = zero(Complex{T})
  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
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
  
  i1 = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
  i2 = (blockIdx().y-UInt32(1)) * blockDim().y + threadIdx().y
  i3 = (blockIdx().z-UInt32(1)) * blockDim().z + threadIdx().z
  step1 = blockDim().x * gridDim().x
  step2 = blockDim().y * gridDim().y
  step3 = blockDim().z * gridDim().z
  
  while i1 <= dim_a
    while i2 <= dim_b
      while i3 <= dim_c
        @inbounds Lx += 
      end
    end
  end
end




function calculate_angmom_kernel(wfc1,wfc2,points,dim,center,V,out_x,out_y,out_z,n1,n2)
  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
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
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  i1 = (blockIdx().y-1) * blockDim().y + threadIdx().y
  i2 = (blockIdx().z-1) * blockDim().z + threadIdx().z
  
  if i > 1 && i1 > 1 && i2 > 1
    if i < dim_a && i1 < dim_b && i2 < dim_c 
      ddax = (wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2])*dadx
      ddbx = (wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2])*dbdx
      ddcx = (wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1])*dcdx
      ddx = ddax+ddbx+ddcx
      
      dday = (wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2])*dady
      ddby = (wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2])*dbdy
      ddcy = (wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1])*dcdy
      ddy = dday+ddby+ddcy
      
      ddaz = (wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2])*dadz
      ddbz = (wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2])*dbdz
      ddcz = (wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1])*dcdz
      ddz = ddaz+ddbz+ddcz
      
      out_x[i,i1,i2] = CUDAnative.conj(wfc1[i,i1,i2])*-1im*((points[i,i1,i2][2]-center_y)*ddz-(points[i,i1,i2][3]-center_z)*ddy)
      out_y[i,i1,i2] = CUDAnative.conj(wfc1[i,i1,i2])*-1im*((points[i,i1,i2][3]-center_z)*ddx-(points[i,i1,i2][1]-center_x)*ddz)
      out_z[i,i1,i2] = CUDAnative.conj(wfc1[i,i1,i2])*-1im*((points[i,i1,i2][1]-center_x)*ddy-(points[i,i1,i2][2]-center_y)*ddx)
      n1[i,i1,i2]    = CUDAnative.conj(wfc1[i,i1,i2])*wfc1[i,i1,i2]
      n2[i,i1,i2]    = CUDAnative.conj(wfc2[i,i1,i2])*wfc2[i,i1,i2]
    end
  end
  return nothing
end
"Calculates the angular momenta between two wavefunctions, around the atom of the second wavefunction."
function calculate_angmom_gpu(wfc1::CuArray{Complex{T},3},wfc2::CuArray{Complex{T},3},points,center,V,num_threads) where T<:AbstractFloat

  total_threads = min(length(wfc1),num_threads)
  threads_x     = floor(Int, total_threads^(1/3))
  threads_y     = threads_x
  threads_z     = threads_x
  threads       = (threads_x,threads_y,threads_z)
  blocks        = Int.(ceil.(size(wfc1)./threads))
  
  Lx = similar(wfc1)
  Ly = similar(wfc1)
  Lz = similar(wfc1)
  n1 = similar(wfc1)
  n2 = similar(wfc1)
  # Lx_ = similar(wfc1)
  # Ly_ = similar(wfc1)
  # Lz_ = similar(wfc1)
  # n1_ = similar(wfc1)
  # n2_ = similar(wfc1)
  dim = CuArray(Int32[size(wfc1)...])
  @cuda (blocks,threads) calculate_angmom_kernel(wfc1,wfc2,points,dim,center,CuArray(V),Lx,Ly,Lz,n1,n2)
  # gpu_reduce(+,Lx,Lx_)
  # gpu_reduce(+,Ly,Ly_)
  # gpu_reduce(+,Lz,Lz_)
  # gpu_reduce(+,n1,n1_)
  # gpu_reduce(+,n2,n2_)
  # n= sqrt(Array(n1)[1]*Array(n2)[1])
  n= sqrt(sum(Array(n1))*sum(Array(n2)))
  return (sum(Array(Lx))/n,sum(Array(Ly))/n,sum(Array(Lz))/n)
  # return (Array(Lx)[1]/n,Array(Ly)[1]/n,Array(Lz)[1]/n)
end

function calculate_angmom_gpu(wfc1::Wfc3D{T},wfc2::Wfc3D{T},num_threads) where T<:AbstractFloat
  if wfc1.atom != wfc2.atom
    return zero(Complex{T}),zero(Complex{T}),zero(Complex{T})
  else
    origin = wfc1[1,1,1].p
    a = wfc1[2,1,1].p - origin
    b = wfc1[1,2,1].p - origin
    c = wfc1[1,1,2].p - origin
    V = inv([convert(Array,a) convert(Array,b) convert(Array,c)])
  
    return calculate_angmom_gpu(CuArray([p.w for p in wfc1.points]),CuArray([p.w for p in wfc2.points]),CuArray([(p.p.x,p.p.y,p.p.z) for p in wfc1.points]),CuArray([wfc2.atom.center.x,wfc2.atom.center.y,wfc2.atom.center.z]),V,num_threads)
  end
end

using CuArrays, CUDAnative

xs, ys, zs = CuArray(rand(100024)), CuArray(rand(100024)), CuArray(zeros(100024))

function kernel_vadd(out, a, b)
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  out[i] = a[i] + b[i]
  return
end

@time @cuda (Int(ceil(length(xs)/1024)), 1024) kernel_vadd(zs, xs, ys)

@assert zs == xs + ys