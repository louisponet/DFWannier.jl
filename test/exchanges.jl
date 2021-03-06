nk = (5,5,5)
R  = DFW.Vec3(2, 0, 0)
fermi = 11.4394
n_ωh = 300
n_ωv = 50
ωh = -30.0
ωv = 0.5


job = DFW.DFControl.DFJob(assetfile(""))
set_magnetization!(job, :Ni1 => [0.0,0.0,1.0], :Ni => [0.0,0.0,-1.0])
hami = DFW.readhami(job)

ω_grid = DFW.setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
kpoints = DFW.ExchangeKGrid(hami, DFW.uniform_kgrid(nk...), R)

@test isapprox(sum(sum.(kpoints.hamiltonian_kgrid.eigvals)), 28310.156014291606)
@test isapprox(sum(kpoints.D),0.07393119748780791 - 1.0312022655727505e-15im)

g_caches = [fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64)) for i=1:3]
G        =fill!(similar(kpoints.hamiltonian_kgrid.eigvecs[1]), zero(ComplexF64))
fill!(G, zero(ComplexF64))
DFW.integrate_Gk!(G, ω_grid[1], fermi, kpoints, g_caches);
exch     = calc_exchanges(hami, atoms(job), fermi; R=R, site_diagonal=false, nk=nk, n_ωh = n_ωh, n_ωv = n_ωv, ωh = ωh, ωv = ωv )
maxJ = maximum([tr(e.J) for e in exch])
@test isapprox(maxJ, 26.044428709929104)

exch1    = calc_exchanges(hami, atoms(job), fermi; R=R, site_diagonal=true, nk=nk, n_ωh = n_ωh, n_ωv = n_ωv, ωh = ωh, ωv = ωv )
maxJ1 = maximum([sum(e.J) for e in exch1])
@test isapprox(maxJ, maxJ1)  

hami_soc = [DFW.TbBlock(b.R_cryst, b.R_cart, DFW.NonColinMatrix(DFW.up(b.block), DFW.down(b.block)), DFW.NonColinMatrix(DFW.up(b.tb_block), DFW.down(b.tb_block))) for b in hami]

set_projections!(job.structure, :Ni1 => [:d], :Ni2 => [:d], :O => [:p], soc=true)
exch     = calc_exchanges(hami_soc, atoms(job), fermi; R=R, site_diagonal=false, nk=nk, n_ωh = n_ωh, n_ωv = n_ωv, ωh = ωh, ωv = ωv )
maxJ1 = maximum([tr(e.J) for e in exch])
@test isapprox(maxJ1, maxJ)
