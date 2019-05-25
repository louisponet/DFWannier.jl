import DFWannier: Point3, SVector

import DFWannier: WannierFunction

x_range = -1:0.01:1
y_range = -1:0.01:1
z_range = -1:0.01:1

wfc_grid = [Point3(x, y, z) for x in x_range, y in y_range, z in z_range]


px_orb = normalize(WannierFunction(wfc_grid, [SVector(((p[1] + 0im) * ℯ^(-norm(p)^2)),) for p in wfc_grid]))

px_orb2 = px_orb'
LinearAlgebra.adjoint!(px_orb2, px_orb2)

@test values(px_orb2) == values(px_orb)

@test norm(px_orb) ≈ norm(px_orb2) ≈ 1.0

py_orb = normalize(WannierFunction(wfc_grid, [SVector(((p[2] + 0im) * ℯ^(-norm(p)^2)),) for p in wfc_grid]))

pz_orb = normalize(WannierFunction(wfc_grid, [SVector(((p[3] + 0im) * ℯ^(-norm(p)^2)),) for p in wfc_grid]))

@test dot(px_orb, py_orb) <= 1.0e-15
@test dot(px_orb, px_orb) ≈ 1.0

Lx = zeros(ComplexF64, 3, 3)
Ly = zeros(ComplexF64, 3, 3)
Lz = zeros(ComplexF64, 3, 3)
for (i1, p1) in enumerate((px_orb, py_orb, pz_orb)), (i2, p2) in enumerate((px_orb, py_orb, pz_orb))
	Lx[i1, i2], Ly[i1, i2], Lz[i1, i2] = DFW.calc_angmom(p1, p2, zero(Point3))
end

@test norm(sum(Lx .- [0 0 0; 0 0 -im; 0 im 0])) < 1e-4
@test norm(sum(Ly .- [0 0 im; 0 0 0; -im 0 0])) < 1e-4
@test norm(sum(Lz .- [0 -im 0; im 0 0; 0 0 0])) < 1e-4

px_orb_up = normalize(WannierFunction(wfc_grid, [SVector((p[1] + 0im, zero(ComplexF64)) .* ℯ^(-norm(p)^2)) for p in wfc_grid]))
px_orb_dn = normalize(WannierFunction(wfc_grid, [SVector((zero(ComplexF64), p[1] + 0im) .* ℯ^(-norm(p)^2)) for p in wfc_grid]))

@test dot(px_orb_dn, px_orb_up) ≈ 0.0
@test DFW.calc_spin(px_orb_up, px_orb_up) ≈ DFW.Point3(0.0 + 0im, 0.0+0.0im, 0.5 + 0.0im)

@test norm(DFW.calc_dip(px_orb, py_orb)) < 1e-17
@test norm(DFW.calc_dip(px_orb_up, px_orb_dn)) < 1e-17


