using DFWannier

T = Float32
x = WannierModel{T}("/Users/ponet/Documents/Fysica/PhD/GeTe/colin/paperxsf/test1","/Users/ponet/Documents/Fysica/PhD/GeTe/fullrel/GeTe_bands.out",[[PhysAtom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[PhysAtom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);
@time tbbandssoc = calculate_eig_cm_angmom_soc(x,90:0.2:110);
@code_warntype read_xsf_file("/Users/ponet/Documents/Fysica/PhD/GeTe/colin/paperxsf/test1")
