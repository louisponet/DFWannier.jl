nk = (5,5,5)
R  = DFW.Vec3(2, 0, 0)
fermi = 13.129
n_ωh = 300
n_ωv = 50
ωh = -30.0
ωv = 0.5

i, structure = DFW.DFControl.wan_read_input(assetfile("wanup.win"))
hami = DFW.read_colin_hami(assetfile("wanup_hr.dat"), assetfile("wandn_hr.dat"), assetfile("wanup_wsvec.dat"), assetfile("wandn_wsvec.dat"), structure)

ω_grid = DFW.setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
kpoints = DFW.ExchangeKGrid(hami, DFW.uniform_shifted_kgrid(nk...), R)

@test isapprox(sum(sum.(sum.(kpoints.hamiltonian_kgrid.eigvecs))), -14.6966233042585 - 3.805745562455212im) 
@test isapprox(sum(sum.(kpoints.hamiltonian_kgrid.eigvals)), 31261.304749999992)
@test isapprox(sum(sum.(kpoints.D)), 0.09113400000005685 + 1.518873858269757e-15im)

g_caches = [fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64)) for i=1:3]
G        =fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64))
fill!(G, zero(ComplexF64))
DFW.integrate_Gk!(G, ω_grid[1], fermi, kpoints, g_caches);

@test isapprox(sum(G[1:16,1:16]), -2.6165384154946817e-5 - 8.046059332175577e-5im)

exch     = calc_exchanges(hami, atoms(structure), fermi; R=R, site_diagonal=false)
maxJ = maximum([tr(e.J) for e in exch])
@test isapprox(maxJ, 21.909887078010946)

exch1    = calc_exchanges(hami, atoms(structure), fermi; R=R, site_diagonal=true)
maxJ1 = maximum([sum(e.J) for e in exch1])
@test isapprox(maxJ1, 21.909887078010946)
@test isapprox(maxJ, maxJ1)  
exch     = calc_exchanges(hami, atoms(structure), fermi, DFW.Exchange4thOrder, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diagonal=false)
