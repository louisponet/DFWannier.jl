```@meta
EditURL = "<unknown>/src/generate_bandstructure.jl"
```

````@example generate_bandstructure
using DFWannier
using Plots
assets_dir = joinpath(splitdir(pathof(DFWannier))[1], "../test/assets")
````

We can use a .chk and .eig file to construct a tight binding Hamiltonian
in the Wannier basis.

````@example generate_bandstructure
hami = read_hamiltonian(joinpath(assets_dir, "Fe/Fe.chk"), joinpath(assets_dir, "Fe/Fe.eig"))
````

We can the generate the bandstructure by first defining a k-path and then perform the
interpolation.

````@example generate_bandstructure
structure = read_w90_input(joinpath(assets_dir, "Fe/Fe.win")).structure
````

First we create some high symmetry kpoints
then we explicitely interpolate between the high symmetry kpoints to form
`bands_kpoints`.

````@example generate_bandstructure
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
bands = wannierbands(hami, band_kpoints)

plot(bands)
````

We can also construct a colinear Tight Binding Hamiltonian by
reading the outputs of up and down Wannierizations

````@example generate_bandstructure
hami = read_hamiltonian(joinpath(assets_dir, "wanup.chk"),
                        joinpath(assets_dir, "wandn.chk"),
                        joinpath(assets_dir, "wanup.eig"),
                        joinpath(assets_dir, "wandn.eig"))
structure = read_w90_input(joinpath(assets_dir, "wanup.win")).structure
bands = wannierbands(hami, band_kpoints)

plot(bands)
````

