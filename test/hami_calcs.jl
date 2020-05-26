nk = (2,2,2)
R  = DFW.Vec3(2, 0, 0)
fermi = 13.129
n_ωh = 300
n_ωv = 50
ωh = -30.0
ωv = 0.5

i, structure = DFW.DFControl.wan_read_input(assetfile("wanup.win"))
hami = readhami(assetfile("wanup_hr.dat"), assetfile("wanup_wsvec.dat"), structure)

wbands = wannierbands(hami, uniform_kgrid(nk...))
@test isapprox(sum(sum.(wbands[3].eigvec)), 0.5674222220608227 - 0.2689932756449306im)


