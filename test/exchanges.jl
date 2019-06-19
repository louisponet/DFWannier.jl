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
hami = DFW.read_colin_hamis(assetfile("wan_up_hr.dat"), assetfile("wan_dn_hr.dat"), structure)


k_grid = DFW.uniform_shifted_kgrid(nk...)
ω_grid = DFW.setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
Hvecs, Hvals, D = DFW.DHvecvals(hami, k_grid)

@test isapprox(sum(sum.(sum.(Hvecs))), 146.3532485358504 - 29.46182528401771im) 
@test isapprox(sum(sum.(Hvals)), 27844.595124999996)
@test isapprox(sum(sum.(D)),-0.010799000000018138 + 1.972617761410861e-15im)

g_caches = [fill!(similar(Hvecs[1]), zero(ComplexF64)) for i=1:3]
G        =fill!(similar(Hvecs[1]), zero(ComplexF64))
fill!(G, zero(ComplexF64))
DFW.integrate_Gk!(G, ω_grid[1], fermi, Hvecs, Hvals, R, k_grid, g_caches);

@test isapprox(sum(G[1:16,1:16]), -1.3067613767591695e-5 + 1.731494091438631e-7im)

exch     = DFW.calc_exchanges(hami, atoms(structure), fermi, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diag=false)
maxJ = abs(maximum(tr.([e.J for e in exch])))
@test maxJ - 43.27722 < 5.0

exch1     = DFW.calc_exchanges(hami, atoms(structure), fermi, R=R, n_ωv = n_ωv, nk=nk, n_ωh=n_ωh, ωv = 0.5, site_diag=true)
maxJ1 = abs(maximum(sum.([e.J for e in exch1])))
@test maxJ1 - 43.27722 < 5.0
@test isapprox(maxJ, maxJ1)  
