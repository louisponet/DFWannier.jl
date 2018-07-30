const workdir = "/home/ponet/Documents/PhD/GeTe/NSOC/"
filename(file) = joinpath(workdir, file)
using LinearAlgebra
using DFWannier
using Plots

job = DFJob(workdir)

hamis=readhamis(job)[1]
λGe = -0.155854;
λTe =  0.318205;
add_wan_data(job)

DFWannier.setsoc!(job, :Ge=> λGe, :Te => λTe);
using GeometryTypes
k_points = [Vec3(i,i,i) for i =0:0.01:0.5]
wanbands = calc_observables(job.structure, k_points, true);
plot(wanbands)
