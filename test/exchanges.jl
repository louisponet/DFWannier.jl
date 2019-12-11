nk = (5,5,5)
R  = DFW.Vec3(-2, 0, 0)
fermi = 11.4394
n_ωh = 300
n_ωv = 50
ωh = -30.0
ωv = 0.5

i, structure = DFW.DFControl.wan_read_input(assetfile("wanup.win"))
DFW.DFControl.projections(structure)
setprojections!(structure, :Ni1=>[:s, :d], :Ni2=>[:s, :d], :O=>[:p])
hami = DFW.read_colin_hami(assetfile("wanup_hr.dat"), assetfile("wandn_hr.dat"), assetfile("wanup_wsvec.dat"), assetfile("wandn_wsvec.dat"), structure)

ω_grid = DFW.setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
kpoints = DFW.ExchangeKGrid(hami, DFW.uniform_shifted_kgrid(nk...), R)

@test isapprox(sum(sum.(sum.(kpoints.hamiltonian_kgrid.eigvecs))), -27.762940492261038 - 25.58983531163306im ) 
@test isapprox(sum(sum.(kpoints.hamiltonian_kgrid.eigvals)), 37498.38612499999)
@test isapprox(sum(sum.(kpoints.D)), 0.37406499999997317 - 7.527317527969424e-16im)

g_caches = [fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64)) for i=1:3]
G        =fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64))
fill!(G, zero(ComplexF64))
DFW.integrate_Gk!(G, ω_grid[1], fermi, kpoints, g_caches);

@test isapprox(sum(G[1:16,1:16]), 0.00025152672795660945 + 3.4781410047963377e-7im)

exch     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange2ndOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=false)
maxJ = abs(maximum(tr.([e.J for e in exch])))
@test maxJ - 43.27722 < 5.0

@show maxJ

exch1     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange2ndOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=true)
maxJ1 = abs(maximum(sum.([e.J for e in exch1])))
@test maxJ1 - 43.27722 < 5.0
@test isapprox(maxJ, maxJ1)  
exch     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange4thOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=false)
