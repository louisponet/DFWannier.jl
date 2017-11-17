using DFWannier
T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-2*0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,2*0.318205]...) for i=1:4]...],true);
x2 = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);
using Plots
@time test = calculate_eig_angmom_soc_bloch_gpu(x,90:0.1:110.);
@time test1 = calculate_eig_angmom_soc_bloch_gpu(x,90:0.1:110.);
@time test1 = calculate_eig_angmom_soc_bloch(x2,90:0.1:110.);
plot(plot(test1[8],:angmom2_x),plot(test[8],:angmom2_x))

@time benchmark = construct_bloch_sum_gpu(x.wfcs[1],x.k_points[1]);

begin 
test1 = construct_bloch_sum_gpu(x.wfcs[1],x.k_points[1])
assert(Array(test1.values)==Array(benchmark.values))
end

test2 = calculate_eig_angmom_soc_bloch(x2,90:110.);
using Plots
test = construct_bloch_sum_gpu(x.wfcs[1],x.k_points[1])
Array(test.values)
test = construct_bloch_sum(x2.wfcs[1],x2.k_points[1])
T=Float32
test_wfc1 =  DFWannier.host2gpu(read_xsf_file("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/wan_00003.xsf",PhysAtom{T}(0.0,0.0,0.1,0.1),T))
test_wfc2=  DFWannier.host2gpu(read_xsf_file("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/wan_00004.xsf",PhysAtom{T}(0.0,0.0,0.1,0.1),T))
dims = size(test_wfc1.values)
cu_dims = CuArray(UInt32[dims...])
Lx = CuArray(zeros(Complex{T},dims))
Ly = CuArray(zeros(Complex{T},dims))
Lz = CuArray(zeros(Complex{T},dims))
n1 = CuArray(zeros(Complex{T},dims))
n2 = CuArray(zeros(Complex{T},dims))
grid = Array(test_wfc1.grid)
origin = grid[1,1,1]
a = grid[2,1,1] .- origin
b = grid[1,2,1] .- origin
c = grid[1,1,2] .- origin
V = CuArray(inv([[a...] [b...] [c...]]))
begin
  # indices = Array{Tuple{Tuple{Int32,Int32,Int32},Tuple{Int32,Int32,Int32}},1}()
  # coeff = Array{Complex{T},1}()
  indices = Array{Tuple{CuArray{Int32,1},CuArray{Int32,1}},1}()
  coeff = Array{CuArray{Complex{T},1},1}()
  k=T[0.2,0.2,0.3]
  for R1=-1:0,R2=-1:0,R3=-1:0
    if R1+R2+R3 == 0
      continue
    end
    R= R1*test_wfc1.cell[1]+R2*test_wfc1.cell[2]+R3*test_wfc1.cell[3]
    ind1,ind2 = DFWannier.find_start(test_wfc1,R,27)
    # push!(indices,((Int32(ind1[1]),Int32(ind1[2]),Int32(ind1[3])),(Int32(ind1[1]-ind2[1]),Int32(ind1[2]-ind2[2]),Int32(ind1[3]-ind2[3]))))
    push!(indices,(CuArray{Int32}([ind1...]),CuArray{Int32}([ind2...])))
    # push!(coeff,Complex{T}(exp(dot(-2*pi*k,[R1,R2,R3])*1im)))
    push!(coeff,CuArray(Complex{T}(exp(dot(-2*pi*k,[R1,R2,R3])*1im))))
  end
  println(length(indices),length(coeff))
  k_wfcs = Array{Wfc3D_gpu{T},1}()
  #optimize so we dont redo the zero index ...
  for wfc in [test_wfc1,test_wfc2]
    push!(k_wfcs,Wfc3D_gpu(wfc.grid,CuArray(zeros(Complex{T},size(wfc.values))),wfc.cell,wfc.atom))
  end
end
@time  for i =1:2000
  construct_bloch_sums_gpu([test_wfc1,test_wfc2],k_wfcs,k,CuArray(indices),CuArray(coeff))
end
calculate_angmom(test_wfc1,test_wfc2,V,CuArray([test_wfc1.atom.center.x,test_wfc1.atom.center.y,test_wfc1.atom.center.z]),dims,Lx,Ly,Lz,n2,n2)

using CUDAnative, CUDAdrv


function haversine_cpu(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32, radius::Float32)
  c1 = cospi(lat1 / 180.0f0)
  c2 = cospi(lat2 / 180.0f0)
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  d1 = sinpi(dlat / 360.0f0)
  d2 = sinpi(dlon / 360.0f0)
  t = d2 * d2 * c1 * c2
  a = d1 * d1 + t
  c = 2.0f0 * asin(min(1.0f0, sqrt(a)))
  return radius * c
end

function pairwise_dist_cpu(lat::Vector{Float32}, lon::Vector{Float32})
  # allocate
  n = length(lat)
  rowresult = Array{Float32}(n, n)
  
  # brute force fill in each cell
  for i in 1:n, j in 1:n
    @inbounds rowresult[i, j] = haversine_cpu(lat[i], lon[i], lat[j], lon[j] , 6372.8f0)
  end
  
  return rowresult    
end

# from https://devblogs.nvidia.com/parallelforall/fast-great-circle-distance-calculation-cuda-c/
function haversine_gpu(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32, radius::Float32)
  # XXX: need to prefix math intrinsics with CUDAnative
  c1 = CUDAnative.cospi(lat1 / 180.0f0)
  c2 = CUDAnative.cospi(lat2 / 180.0f0)
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  d1 = CUDAnative.sinpi(dlat / 360.0f0)
  d2 = CUDAnative.sinpi(dlon / 360.0f0)
  t = d2 * d2 * c1 * c2
  a = d1 * d1 + t
  c = 2.0f0 * CUDAnative.asin(CUDAnative.min(1.0f0, CUDAnative.sqrt(a)))
  return radius * c
end

# pairwise distance calculation kernel
function pairwise_dist_kernel(lat::CuDeviceVector{Float32}, lon::CuDeviceVector{Float32},
  rowresult::CuDeviceMatrix{Float32}, n)
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  j = (blockIdx().y-1) * blockDim().y + threadIdx().y
  
  if i <= n && j <= n
    # store to shared memory
    shmem = @cuDynamicSharedMem(Float32, 2*blockDim().x + 2*blockDim().y)
    if threadIdx().y == 1
      shmem[threadIdx().x] = lat[i]
      shmem[blockDim().x + threadIdx().x] = lon[i]
    end
    if threadIdx().x == 1
      shmem[2*blockDim().x + threadIdx().y] = lat[j]
      shmem[2*blockDim().x + blockDim().y + threadIdx().y] = lon[j]
    end
    sync_threads()
    
    # load from shared memory
    lat_i = shmem[threadIdx().x]
    lon_i = shmem[blockDim().x + threadIdx().x]
    lat_j = shmem[2*blockDim().x + threadIdx().y]
    lon_j = shmem[2*blockDim().x + blockDim().y + threadIdx().y]
    
    @inbounds rowresult[i, j] = haversine_gpu(lat_i, lon_i, lat_j, lon_j, 6372.8f0)
  end
end

function pairwise_dist_gpu(lat::Vector{Float32}, lon::Vector{Float32})
  # upload
  lat_gpu = CuArray(lat)
  lon_gpu = CuArray(lon)
  
  # allocate
  n = length(lat)
  rowresult_gpu = CuArray{Float32}(n, n)
  
  # calculate launch configuration
  # NOTE: we want our launch configuration to be as square as possible,
  #       because that minimizes shared memory usage
  ctx = CuCurrentContext()
  dev = device(ctx)
  total_threads = min(n, attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))
  threads_x = floor(Int, sqrt(total_threads))
  threads_y = total_threads ÷ threads_x
  threads = (threads_x, threads_y)
  blocks = ceil.(Int, n ./ threads)
  
  # calculate size of dynamic shared memory
  shmem = 2 * sum(threads) * sizeof(Float32)
  println(shmem)
  @cuda (blocks, threads, shmem) pairwise_dist_kernel(lat_gpu, lon_gpu, rowresult_gpu, n)
  
  return Array(rowresult_gpu)
end


# generate reasonable data
const n = 10000
const lat = rand(Float32, n) .* 45
const lon = rand(Float32, n) .* -120

using Compat
using Compat.Test

@test pairwise_dist_cpu(lat, lon) ≈ pairwise_dist_gpu(lat, lon)