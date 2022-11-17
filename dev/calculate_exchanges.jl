using LinearAlgebra#hide
BLAS.set_num_threads(1)#hide
using DFWannier
assets_dir = joinpath(splitdir(pathof(DFWannier))[1], "../test/assets")

# We first read the colinear Hamiltonian from the outputs of wannier90.
hami = read_hamiltonian(joinpath(assets_dir, "wanup.chk"),
                        joinpath(assets_dir, "wandn.chk"),
                        joinpath(assets_dir, "wanup.eig"),
                        joinpath(assets_dir, "wandn.eig"))

# We can the generate the bandstructure by first defining a k-path and then perform the
# interpolation.
structure = read_w90_input(joinpath(assets_dir, "wanup.win")).structure
# First we create some high symmetry kpoints
# then we explicitely interpolate between the high symmetry kpoints to form
# `bands_kpoints`.
kpoints = [Vec3(0.0, 0.0, 0.5),
           Vec3(0.0, 0.5, 0.5),
           Vec3(0.5, 0.5, 0.5),
           Vec3(0.5, 0.5, 0.0),
           Vec3(0.5, 0.0, 0.0),
           Vec3(0.0, 0.0, 0.0)]
band_kpoints = eltype(kpoints)[] 
for i = 1:length(kpoints)-1
    for α in range(0, 1, 20)
        push!(band_kpoints, Vec3((1-α) .* kpoints[i] .+ α .* kpoints[i+1]))
    end
end

# In order to calculate the magnetic exchanges we need to specify the fermi level (e.g. can be found in an nscf output file),
# and we need to specify the atoms we want to calculate the exchanges between.
# We set the number of k points used for the kpoint interpolation, and number of frequency points to calculate the
# contour integral (`n_ωh`, `n_ωv`).
exch = calc_exchanges(hami, structure[element(:Ni)], 12.0; nk=(5,5,5), n_ωh = 300, n_ωv = 30)
# This leads to a list of exchanges where each holds the J matrix, whose trace is the actual exchange between the sites specified
# by `atom1` and `atom2`.

# To calculate the exchange between the atoms in the central unit cell and those in a shifted one we can use R.
# In this specific case we are calculating the exchanges towards the unit cell shifted twice along the `b` cell vector.
exch = calc_exchanges(hami, structure[element(:Ni)], 12.0, R=(0,2,0); nk=(5,5,5), n_ωh = 300, n_ωv = 30)
