using DFWannier
T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);

using CUDAdrv, CUDAnative
function bloch_kernel(wfc_orig, indices,coefficients,cell,dim, out)
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
    if i1 <= dim_a && i1>= ind1[1] &&  j1 <= dim_a && j1 > 0
      if i2 <= dim_b && i2>= ind1[2] &&  j2 <= dim_b && j2 > 0
        if i3 <= dim_c && i3>= ind1[3] &&  j3 <= dim_c && j3 > 0
          out[i1,i2,i3] += c*wfc_orig[j1,j2,j3]
        end
      end
    end
  end 
  return nothing
end

function construct_bloch_sum_gpu(wfc::Wfc3D{T}, k::Array) where T <: AbstractFloat

  indices = Array{Tuple{Tuple{Int32,Int32,Int32},Tuple{Int32,Int32,Int32}},1}()
  coefficients = Array{Complex{T},1}()
  for R1=-1:1,R2=-1:1,R3=-1:1
    R= R1*wfc.cell[1]+R2*wfc.cell[2]+R3*wfc.cell[3]
    ind1,ind2 = DFWannier.find_start(wfc,R,27)
    push!(indices,((Int32(ind1[1]),Int32(ind1[2]),Int32(ind1[3])),(Int32(ind1[1]-ind2[1]),Int32(ind1[2]-ind2[2]),Int32(ind1[3]-ind2[3]))))
    push!(coefficients,Complex{Float32}(exp(dot(-2*pi*k,[R1,R2,R3])*1im)))
  end

  dev = CuDevice(0)
  ctx = CuContext(dev)
  points = CuArray([p.w for p in wfc.points])
  out = similar(points)

  total_threads = min(length(wfc.points),attribute(dev,CUDAdrv.MAX_THREADS_PER_BLOCK))
  threads_x     = floor(Int, total_threads^(1/3))
  threads_y     = threads_x
  threads_z     = threads_x
  threads       = (threads_x,threads_y,threads_z)
  blocks        = Int.(ceil.(size(wfc.points)./threads))
  println(blocks)
  @time @cuda (blocks,threads) bloch_kernel(points,CuArray(indices),CuArray(coefficients),CuArray(wfc.cell),CuArray(Int64[size(points)...]),out)
  t_points = Array(out)
  destroy!(ctx)
  t_out    = Array{WfcPoint3D{T},3}(size(t_points))
  for i in eachindex(t_points)
    t_out[i] = WfcPoint3D(t_points[i],wfc.points[i].p)
  end
  return Wfc3D(t_out,wfc.cell,wfc.atom)
end

@time test = construct_bloch_sum_gpu(x.wfcs[1],x.k_points[1])
write_xsf_file("/home/ponet/Downloads/test.xsf",test)
@time test1= DFWannier.construct_bloch_sum(x.wfcs[1],x.k_points[1])
test1.points[3]
test.points[3]