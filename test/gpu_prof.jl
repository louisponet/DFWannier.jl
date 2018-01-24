#!/opt/julia0.6/julia
using DFWannier
using CUDAnative
using CuArrays 
CUDAnative.@profile begin
  T=Float64
  test_wfc1 =  DFWannier.host2gpu(read_xsf_file("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/wan_00003.xsf",Atom{T}(0.0,0.0,0.1,0.1),T))
  test_wfc2=  DFWannier.host2gpu(read_xsf_file("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/wan_00004.xsf",Atom{T}(0.0,0.0,0.1,0.1),T))
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
  indices = Array{Tuple{Tuple{Int32,Int32,Int32},Tuple{Int32,Int32,Int32}},1}()
  t_coeff = Array{Complex{T}}(27)
  t=1
  k=T[0.2,0.2,0.3]
  for R1=-1:1,R2=-1:1,R3=-1:1
    R= R1*test_wfc1.cell[1]+R2*test_wfc1.cell[2]+R3*test_wfc1.cell[3]
    ind1,ind2 = DFWannier.find_start(test_wfc1,R,27)
    push!(indices,((Int32(ind1[1]),Int32(ind1[2]),Int32(ind1[3])),(Int32(ind1[1]-ind2[1]),Int32(ind1[2]-ind2[2]),Int32(ind1[3]-ind2[3]))))
    t_coeff[t] = Complex{T}(exp(dot(-2*pi*k,[R1,R2,R3])*1im))
    t+=1
  end
  coefficients = CuArray(t_coeff)
  cu_indices = CuArray(indices)
  k_wfcs = Array{Wfc3D_gpu{T},1}(2)
  for (n,wfc) in enumerate([test_wfc1,test_wfc2])
    k_wfcs[n] = Wfc3D_gpu(wfc.grid,CuArray(zeros(Complex{T},size(wfc.values))),wfc.cell,wfc.atom)
  end
  # DFWannier.construct_bloch_sums([test_wfc1,test_wfc2],k_wfcs,k,cu_indices,coefficients)
  begin
  correct = calculate_angmom(test_wfc1,test_wfc2,V,CuArray([test_wfc1.atom.center.x,test_wfc1.atom.center.y,test_wfc1.atom.center.z]),dims,Lx,Ly,Lz,n2,n2)
  test = calculate_angmom(test_wfc1,test_wfc2,V,CuArray([test_wfc1.atom.center.x,test_wfc1.atom.center.y,test_wfc1.atom.center.z]),dims,Lx,Ly,Lz,n2,n2)
  @time for i =1:2000
    test = calculate_angmom(test_wfc1,test_wfc2,V,CuArray([test_wfc1.atom.center.x,test_wfc1.atom.center.y,test_wfc1.atom.center.z]),dims,Lx,Ly,Lz,n2,n2)

  end
  assert(test==correct)
  end
end
  
  