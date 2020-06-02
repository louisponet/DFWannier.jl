nk = (2,2,2)
R  = DFW.Vec3(2, 0, 0)
fermi = 13.129
n_ωh = 300
n_ωv = 50
ωh = -30.0
ωv = 0.5

i, structure = DFW.DFControl.wan_read_input(assetfile("wanup.win"))
hami = readhami(DFW.read_chk(assetfile("wanup.chk")), assetfile("wanup.eig"))

wbands = wannierbands(hami, uniform_kgrid(nk...))
@test isapprox(sum(wbands[3].eigvals), 26.2642711779676)
