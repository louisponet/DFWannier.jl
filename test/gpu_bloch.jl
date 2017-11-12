using DFWannier
T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);
using CUDAdrv
CUDAdrv.version()
dev = CuDevice(0)
ctx = CuContext(dev)
test= calculate_angmom_gpu(x.wfcs[1],x.wfcs[2],1024)
test1= calculate_angmom(x.wfcs[1],x.wfcs[2])
bands = calculate_eig_angmom_bloch_gpu(x,x.k_points)
test = calculate_eig_angmom_bloch(x,x.k_points)
