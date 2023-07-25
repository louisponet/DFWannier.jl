var documenterSearchIndex = {"docs":
[{"location":"exchanges.html#Exchanges","page":"Exchanges","title":"Exchanges","text":"","category":"section"},{"location":"exchanges.html","page":"Exchanges","title":"Exchanges","text":"CurrentModule = DFWannier","category":"page"},{"location":"exchanges.html","page":"Exchanges","title":"Exchanges","text":"Using the TBHamiltonian and calc_exchanges, it is possible to calculate the magnetic exchange parameters J_ij between atoms i and j for the isotropic Heisenberg model:\\\nE = sum_ij J_ij overrightarrowS_i cdot overrightarrowS_j","category":"page"},{"location":"exchanges.html","page":"Exchanges","title":"Exchanges","text":"This involves calculating the Green's functions G and on-site magnetic field matrices Delta, which then determine J as J_{ij} = \\frac{1}{2\\pi} \\int{-\\infty}^{Ef} d\\varepsilon \\Deltai G{ij}^{\\downarrow}(\\varepsilon) \\Deltaj G{ji}^{\\uparrow}(\\varepsilon). See Weak ferromagnetism in antiferromagnets: Fe2O3 and La2CuO4.","category":"page"},{"location":"exchanges.html","page":"Exchanges","title":"Exchanges","text":"Exchange2ndOrder\nExchange4thOrder\ncalc_exchanges","category":"page"},{"location":"exchanges.html#DFWannier.Exchange2ndOrder","page":"Exchanges","title":"DFWannier.Exchange2ndOrder","text":"Exchange2ndOrder{T <: AbstractFloat}\n\nThis holds the exhanges between different orbitals and calculated sites. Projections and atom datablocks are to be found in the corresponding wannier input file. It turns out the ordering is first projections, then atom order in the atoms datablock.\n\n\n\n\n\n","category":"type"},{"location":"exchanges.html#DFWannier.Exchange4thOrder","page":"Exchanges","title":"DFWannier.Exchange4thOrder","text":"Exchange4thOrder{T <: AbstractFloat}\n\nThis holds the exhanges between different orbitals and calculated sites. Projections and atom datablocks are to be found in the corresponding wannier input file. It turns out the ordering is first projections, then atom order in the atoms datablock.\n\n\n\n\n\n","category":"type"},{"location":"exchanges.html#DFWannier.calc_exchanges","page":"Exchanges","title":"DFWannier.calc_exchanges","text":"calc_exchanges(hamiltonian::TBHamiltonian, atoms::Vector{<:Atom}, fermi, exchange_type; kwargs...)\n\nCalculates the magnetic exchange parameters between the atoms. exchange_type can be Exchange2ndOrder or Exchange4thOrder. The kwargs control various numerical parameters for the calculation:\n\nnk = (10,10,10): the amount of k-points to be used for the uniform interpolation grid.\nR = (0,0,0): the unit cell index to which the exchange parameters are calculated.\nωh = -30.0: the lower bound of the energy integration\nωv = 0.15: the height of the contour in complex space to integrate the Green's functions\nn_ωh = 3000: number of integration points along the horizontal contour direction\nn_ωv = 500: number of integration points along the vertical contour direction\nsite_diagonal = false: if true the hamiltonians and Δ will diagonalized on-site and the\n\nreturned exchange matrices hold the exchanges between well-defined orbitals. If this is not done, the exchange matrices entries don't mean anything on themselves and a trace should be performed to find the exchange between the spins on sites i and j.\n\n\n\n\n\n","category":"function"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"EditURL = \"<unknown>/src/calculate_exchanges.jl\"","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"using LinearAlgebra#hide\nBLAS.set_num_threads(1)#hide\nusing DFWannier\nassets_dir = joinpath(splitdir(pathof(DFWannier))[1], \"../test/assets\")","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"We first read the colinear Hamiltonian from the outputs of wannier90.","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"hami = read_hamiltonian(joinpath(assets_dir, \"wanup.chk\"),\n                        joinpath(assets_dir, \"wandn.chk\"),\n                        joinpath(assets_dir, \"wanup.eig\"),\n                        joinpath(assets_dir, \"wandn.eig\"))","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"We can the generate the bandstructure by first defining a k-path and then perform the interpolation.","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"structure = read_w90_input(joinpath(assets_dir, \"wanup.win\")).structure","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"First we create some high symmetry kpoints then we explicitely interpolate between the high symmetry kpoints to form bands_kpoints.","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"kpoints = [Vec3(0.0, 0.0, 0.5),\n           Vec3(0.0, 0.5, 0.5),\n           Vec3(0.5, 0.5, 0.5),\n           Vec3(0.5, 0.5, 0.0),\n           Vec3(0.5, 0.0, 0.0),\n           Vec3(0.0, 0.0, 0.0)]\nband_kpoints = eltype(kpoints)[]\nfor i = 1:length(kpoints)-1\n    for α in range(0, 1, 20)\n        push!(band_kpoints, Vec3((1-α) .* kpoints[i] .+ α .* kpoints[i+1]))\n    end\nend","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"In order to calculate the magnetic exchanges we need to specify the fermi level (e.g. can be found in an nscf output file), and we need to specify the atoms we want to calculate the exchanges between. We set the number of k points used for the kpoint interpolation, and number of frequency points to calculate the contour integral (n_ωh, n_ωv).","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"exch = calc_exchanges(hami, structure[element(:Ni)], 12.0; nk=(5,5,5), n_ωh = 300, n_ωv = 30)","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"This leads to a list of exchanges where each holds the J matrix, whose trace is the actual exchange between the sites specified by atom1 and atom2.","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"To calculate the exchange between the atoms in the central unit cell and those in a shifted one we can use R. In this specific case we are calculating the exchanges towards the unit cell shifted twice along the b cell vector.","category":"page"},{"location":"calculate_exchanges.html","page":"-","title":"-","text":"exch = calc_exchanges(hami, structure[element(:Ni)], 12.0, R=(0,2,0); nk=(5,5,5), n_ωh = 300, n_ωv = 30)","category":"page"},{"location":"tight_binding.html#Tight-Binding","page":"Tight Binding","title":"Tight Binding","text":"","category":"section"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"Various tight binding operators can be generated using the outputs of Wannier90.","category":"page"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"CurrentModule = DFWannier","category":"page"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"TBBlock\nTBOperator","category":"page"},{"location":"tight_binding.html#DFWannier.TBBlock","page":"Tight Binding","title":"DFWannier.TBBlock","text":"TBBlock\n\nBuilding block for TBOperator. It holds the matrix elements of the operator between central and a shifted unit cell. Upon construction, the wigner-seitz shifts are taken into account to create the correct matrix elements between the Wannierfunctions, stored in tb_block. The block field is basically tb_block but with each element divided by the amount of Wigner-Seitz degeneracies and shifts which speeds up later k-point interpolation.\n\n\n\n\n\n","category":"type"},{"location":"tight_binding.html#DFWannier.TBOperator","page":"Tight Binding","title":"DFWannier.TBOperator","text":"TBOperator\n\nAlias for a Vector of TBBlocks. Indexing with NTuple{3,Int} or Vec3 is supported which allows for easily retrieving the TBBlock that corresponds to the shifted unit cell. Aliases: TBHamiltonian, TBSpin\n\n\n\n\n\n","category":"type"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"The following functions can be used to generate specific operators.","category":"page"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"read_hamiltonian\nread_spin\nread_r","category":"page"},{"location":"tight_binding.html#DFWannier.read_hamiltonian","page":"Tight Binding","title":"DFWannier.read_hamiltonian","text":"read_hamiltonian(chk::NamedTuple, eigvals::Matrix)\n\nUses the Wannier90 chkpoint info in chk and DFT eigenvals read with [read_eig] to construct the TBHamiltonian.\n\n\n\n\n\nread_hamiltonian(job::Job)\n\nGoes through the job and will attempt to read the hamiltonian files. If it finds a colinear calculation in the job it will read the up and down hamiltonians, if the job was either nonmagnetic or noncolinear it will read only one hamiltonian file (there should be only one).\n\n\n\n\n\n","category":"function"},{"location":"tight_binding.html#DFWannier.read_spin","page":"Tight Binding","title":"DFWannier.read_spin","text":"read_spin(chk_file, spn_file)\nread_spin(job::Job)\n\nReads the .spn and .chk files to generate a TBSpin tight-binding spin operator.\n\n\n\n\n\n","category":"function"},{"location":"tight_binding.html#DFWannier.read_r","page":"Tight Binding","title":"DFWannier.read_r","text":"read_r(chk_file::AbstractString, nnkp_file::AbstractString)\nread_r(job::Job)\n\nConstructs the r [TBOperator] from the Wannier90 .chk and .nnkp files. This requires that the k_neighbor_weights is written into the .chk file and might need a patched Wannier90 version.\n\n\n\n\n\n","category":"function"},{"location":"tight_binding.html#Reciprocal","page":"Tight Binding","title":"Reciprocal","text":"","category":"section"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"The tight binding operators can be used to interpolate properties in reciprocal space.","category":"page"},{"location":"tight_binding.html","page":"Tight Binding","title":"Tight Binding","text":"Hk\nHamiltonianKGrid\nWannierBand\nwannierbands","category":"page"},{"location":"tight_binding.html#DFWannier.Hk","page":"Tight Binding","title":"DFWannier.Hk","text":"Hk(hamiltonian::TBHamiltonian, kpoint::Vec3)\nHk!(hk::AbstractMatrix, hamiltonian::TBHamiltonian, kpoint::Vec3)\n\nConstructs the reciprocal Hamiltonian at a given k-point.  \n\n\n\n\n\n","category":"function"},{"location":"tight_binding.html#DFWannier.HamiltonianKGrid","page":"Tight Binding","title":"DFWannier.HamiltonianKGrid","text":"HamiltonianKGrid(hami::TBHamiltonian{T}, nk, H_function_k::Function = x -> nothing) where T\nHamiltonianKGrid(hami::TBHamiltonian{T}, k_grid, H_function_k::Function = x -> nothing) where T\n\nTakes a k grid, calculates Hk for each of them and diagonalizes. Only the eigenvectors and eigenvalues of Hk are stored, the H_function_k function is called on the intermediate Hk. \n\n\n\n\n\n","category":"type"},{"location":"tight_binding.html#DFWannier.WannierBand","page":"Tight Binding","title":"DFWannier.WannierBand","text":"WannierBand\n\nRepresents a Wannier interpolated band. See also wannierbands. \n\n\n\n\n\n","category":"type"},{"location":"tight_binding.html#DFWannier.wannierbands","page":"Tight Binding","title":"DFWannier.wannierbands","text":"wannierbands(hamiltonian::TBHamiltonian, kpoints::Vector{Vec3})\nwannierbands(hamiltonian::TBHamiltonian, bands::Vector{DFControl.AbstractBand}\n\nConstructs the whole bandstructure for a given set of k-points and TBHamiltonian.\n\n\n\n\n\n","category":"function"},{"location":"berry.html#Berry","page":"Berry","title":"Berry","text":"","category":"section"},{"location":"berry.html","page":"Berry","title":"Berry","text":"Stay tuned","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"EditURL = \"<unknown>/src/generate_bandstructure.jl\"","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"using DFWannier\nusing Plots\nassets_dir = joinpath(splitdir(pathof(DFWannier))[1], \"../test/assets\")","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"We can use a .chk and .eig file to construct a tight binding Hamiltonian in the Wannier basis.","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"hami = read_hamiltonian(joinpath(assets_dir, \"Fe/Fe.chk\"), joinpath(assets_dir, \"Fe/Fe.eig\"))","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"We can the generate the bandstructure by first defining a k-path and then perform the interpolation.","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"structure = read_w90_input(joinpath(assets_dir, \"Fe/Fe.win\")).structure","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"First we create some high symmetry kpoints then we explicitely interpolate between the high symmetry kpoints to form bands_kpoints.","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"kpoints = [Vec3(0.0, 0.0, 0.5),\n           Vec3(0.0, 0.5, 0.5),\n           Vec3(0.5, 0.5, 0.5),\n           Vec3(0.5, 0.5, 0.0),\n           Vec3(0.5, 0.0, 0.0),\n           Vec3(0.0, 0.0, 0.0)]\nband_kpoints = eltype(kpoints)[]\nfor i = 1:length(kpoints)-1\n    for α in range(0, 1, 20)\n        push!(band_kpoints, Vec3((1-α) .* kpoints[i] .+ α .* kpoints[i+1]))\n    end\nend\nbands = wannierbands(hami, band_kpoints)\n\nplot(bands)","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"We can also construct a colinear Tight Binding Hamiltonian by reading the outputs of up and down Wannierizations","category":"page"},{"location":"generate_bandstructure.html","page":"-","title":"-","text":"hami = read_hamiltonian(joinpath(assets_dir, \"wanup.chk\"),\n                        joinpath(assets_dir, \"wandn.chk\"),\n                        joinpath(assets_dir, \"wanup.eig\"),\n                        joinpath(assets_dir, \"wandn.eig\"))\nstructure = read_w90_input(joinpath(assets_dir, \"wanup.win\")).structure\nbands = wannierbands(hami, band_kpoints)\n\nplot(bands)","category":"page"},{"location":"index.html#DFWannier-Documentation","page":"Home","title":"DFWannier Documentation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"This package is meant to be used as a post processing tool for Wannier90. It relies heavily on DFControl. The main capabilities are","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Generating the Tight-Binding Hamiltonian, Spin and dipole operators\nInterpolating bands and other properties in k-space using said operators\nGenerating the real space Wannier functions\nCalculate various Berry and geometric properties\nCalculate the magnetic exchange parameters for the Heisenberg model ","category":"page"}]
}