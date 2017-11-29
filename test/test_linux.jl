using DFWannier
using Plots

T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);

dfbandssoc = read_qe_bands_file("/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",T);
tbbandssoc = calculate_eig_soc(x);
plot(tbbandssoc,dfbandssoc)
plot(tbbandssoc,:eigvals)

tbbandssoc = calculate_eig_cm_angmom_soc(x);
plot(tbbandssoc,[:cm_x,:cm_y,:cm_z])
ENV["TRACE"]=true
using CUDAdrv, CUDAnative
function construct_bloch_sum(wfc::Wfc3D{T}, k::Array) where T <: AbstractFloat
  function bloch_kernel(wfc_orig, indices,coefficients,cell, out)
    dim_a = size(wfc_orig)[1]
    dim_b = size(wfc_orig)[2]
    dim_c = size(wfc_orig)[3]
    i1 = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i2 = (blockIdx().y-1) * blockDim().y + threadIdx().y
    i3 = (blockIdx().z-1) * blockDim().z + threadIdx().z

    for (c,(ind1,ind2)) in zip(coefficients,indices)
      j1 = i1+ind2[1]
      j2 = i2+ind2[2]
      j3 = i3+ind2[3]
      if i1 <= dim_a && i1>= ind1[1] &&  j1 <= dim_a && j1 > 0
        if i2 <= dim_b && i2>= ind1[2] &&  j2 <= dim_b && j2 > 0
          if i3 <= dim_c && i3>= ind1[3] &&  j3 <= dim_c && j3 > 0
            out[i1,i2,i3] += c*wfc_orig[j1,j2,j3]
          end
        end
      end
    end 
  end

  indices = Array{Tuple{Tuple{Int,Int,Int},Tuple{Int,Int,Int}},1}()
  coefficients = Array{Complex{T},1}()
  for R1=-1:1,R2=-1:1,R3=-1:1
    R= R1*wfc.cell[1]+R2*wfc.cell[2]+R3*wfc.cell[3]
    ind1,ind2 = DFWannier.find_start(wfc,R,27)

    push!(indices,(ind1,(ind1[1]-ind2[1],ind1[2]-ind2[2],ind1[3]-ind2[3])))
    push!(coefficients,Complex{T}(exp(dot(-2*pi*k,[R1,R2,R3])*1im)))
  end

  dev = CuDevice(0)
  ctx = CuContext(dev)
  points = CuArray([p.w for p in wfc.points])
  out = similar(points)

  total_threads = min(length(wfc.points),Int(CUDAdrv.MAX_THREADS_PER_BLOCK))
  threads_x     = floor(Int, total_threads^(1/3))
  threads_y     = threads_x
  threads_z     = threads_x
  threads       = (threads_x,threads_y,threads_z)
  blocks = ceil.(Int,length(wfc.points) ./ threads)

  @cuda (blocks,threads) bloch_kernel(points,CuArray(indices),CuArray(coefficients),CuArray(wfc.cell),out)
  return out
end

test = construct_bloch_sum(x.wfcs[1],x.k_points[1])
