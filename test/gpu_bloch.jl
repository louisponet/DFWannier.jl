using DFWannier
T= Float32
x = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...],true);
x2 = WannierModel{T}("/home/ponet/Documents/PhD/GeTe/NSOC/paperxsf/","/home/ponet/Documents/PhD/GeTe/SOC/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);


@time testkkkkk = calculate_angmom(x.wfcs[2],x.wfcs[3])
@time test1 = calculate_angmom(x2.wfcs[2],x2.wfcs[3])
test=construct_bloch_sum_gpu(x.wfcs[2],x.k_points[100])
@time test1 =  calculate_eig_angmom_soc_bloch_gpu(x,90.:0.1:110);
test2 =  calculate_eig_angmom_soc_bloch_gpu(x,90.:0.01:110);
plot(test[8],:angmom2_x)
test1 =  calculate_eig_angmom_soc_bloch(x2,90:0.01:110);



