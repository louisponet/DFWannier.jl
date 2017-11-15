using DFWannier
T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...],true);
x2 = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);
using CUDAnative
using ProfileView
@profile calculate_eig_angmom_soc_bloch_gpu(x,1.:0.1:200);
@time test = calculate_eig_angmom_soc_bloch(x2,1.:0.1:200);
using BenchmarkTools
@benchmark calculate_eig_angmom_soc_bloch_gpu(x,90.:0.1:110)

using ProfileView
Profile.clear()

Base.@profile calculate_eig_angmom_soc_bloch_gpu(x,90.:0.1:110);
ProfileView.view()
using Plots
plot(test[8],:angmom2_x)
test2 =  calculate_eig_angmom_soc_bloch_gpu(x,90.:0.01:110);
plot(test[8],:angmom2_x)
test1 =  calculate_eig_angmom_soc_bloch(x2,90:0.1:110);
plot!(test1[8],:angmom2_x,c=:red)


using CUDAnative,CuArrays
using CUDAdrv:CuContext,CuDevice,MAX_THREADS_PER_BLOCK,destroy!,attribute

@inline function threadID3D()
  i1= (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
  i2 = (blockIdx().y-UInt32(1)) * blockDim().y + threadIdx().y
  i3 = (blockIdx().z-UInt32(1)) * blockDim().z + threadIdx().z
  return i1,i2,i3
end

function dumb_kernel(in::CuDeviceArray{UInt32,3},out::CuDeviceArray{UInt32,3}) where T
  i1,i2,i3 = threadID3D()
  if i1 == UInt32(1)
    out[i1,i2,i3] = in[i1,i2,i3]
  end
  return nothing
end


testarr = rand(Complex{Float32},(64,64,64))
testarr = UInt32[i1+i2+i3 for i1=1:82,i2=1:82,i3=1:82]
test_cuarr = CuArray(testarr)
test_cuarr_o = CuArray(zeros(UInt32,size(testarr)))

num_threads = attribute(CuDevice(0),MAX_THREADS_PER_BLOCK)
total_threads = min(length(testarr),num_threads)
max_threads_per_dim = floor(Int, total_threads^(1/ndims(testarr)))
dims = size(testarr)
threads_x     = cld(dims[1],max_threads_per_dim)
threads_y     = cld(dims[2],max_threads_per_dim)
threads_z     = cld(dims[3],max_threads_per_dim)
threads       = (threads_x,threads_y,threads_z)
blocks        = Int.(ceil.(size(testarr)./threads))
println(blocks,threads)
@cuda ((10,10,10),(9,9,9)) dumb_kernel(test_cuarr,test_cuarr_o)
size(filter(x->!iszero(x),Array(test_cuarr_o)))
first_sum = sum(test_cuarr_o)
@cuda (blocks,threads) dumb_kernel(test_cuarr,test_cuarr_o)
sec_sum   = sum(test_cuarr_o)
println(first_sum)
println(sec_sum)
