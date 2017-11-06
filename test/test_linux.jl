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