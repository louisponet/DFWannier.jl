nk = (5,5,5)
R  = DFW.Vec3(-2, 0, 0)
fermi = 11.7868
n_ωh = 300
n_ωv = 50
ωh = -30.0
ωv = 0.5
i, structure = DFW.DFControl.wan_read_input(assetfile("wan_up.win"))
DFW.DFControl.projections(structure)
setprojections!(structure, :Ni1=>[:d], :Ni2=>[:d], :O=>[:s])
hami = DFW.read_colin_hami(assetfile("wan_up_hr.dat"), assetfile("wan_dn_hr.dat"), structure)


ω_grid = DFW.setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
kpoints = DFW.ExchangeKGrid(hami, DFW.uniform_shifted_kgrid(nk...), R)

@test isapprox(sum(sum.(sum.(kpoints.hamiltonian_kgrid.eigvecs))), 146.3532485358504 - 29.46182528401771im) 
@test isapprox(sum(sum.(kpoints.hamiltonian_kgrid.eigvals)), 27844.595124999996)
@test isapprox(sum(sum.(kpoints.D)),-0.010799000000018138 + 1.972617761410861e-15im)

g_caches = [fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64)) for i=1:3]
G        =fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64))
fill!(G, zero(ComplexF64))
DFW.integrate_Gk!(G, ω_grid[1], fermi, kpoints, g_caches);

@test isapprox(sum(G[1:16,1:16]), -1.3067613767591695e-5 + 1.731494091438631e-7im)

exch     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange2ndOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=false)
maxJ = abs(maximum(tr.([e.J for e in exch])))
@test maxJ - 43.27722 < 5.0


exch1     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange2ndOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=true)
maxJ1 = abs(maximum(sum.([e.J for e in exch1])))
@test maxJ1 - 43.27722 < 5.0
@test isapprox(maxJ, maxJ1)  
exch     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange4thOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=false)
