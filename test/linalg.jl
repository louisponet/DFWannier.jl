for F in (Float32, Float64)
	orig_up = Hermitian(rand(Complex{F}, 25, 25))
	orig_dn = Hermitian(rand(Complex{F}, 25, 25))
	colin = DFW.ColinMatrix(orig_up, orig_dn)
	normal_eig1, normal_eig2 = eigen(orig_up), eigen(orig_dn)
	cache      = DFW.EigCache(colin)
	cached_eig = eigen(colin, cache)
	@test sum(normal_eig1.values) + sum(normal_eig2.values) ≈ sum(cached_eig.values)
	@test Array(normal_eig1) ≈ Array(cached_eig)[1:25, 1:25] ≈ DFW.up(colin)
	@test Array(normal_eig2) ≈ Array(cached_eig)[1:25, 26:50] ≈ DFW.down(colin)
end

for F in (Float32, Float64)
	t = rand(Complex{F}, 50, 50)
	orig       = (t + t')/2
	normal_eig = eigen(orig)
	cache      = DFW.EigCache(orig)
	cached_eig = eigen(orig, cache)
	@test sum(normal_eig.values) ≈ sum(cached_eig.values) 
	@test Array(normal_eig) ≈ Array(cached_eig) ≈ orig
end

mat = rand(20, 20)
atoms = [DFControl.Atom(:Fe, DFControl.element(:Fe), DFControl.Point3(0.0, 0.0,0.0).*DFControl.angstrom, DFControl.Point3(0.0,0.0,0.0), projections=[DFControl.Projection(DFControl.orbital(:d), 1, 10)]), DFControl.Atom(:Fe, DFControl.element(:Fe), DFControl.Point3(1.0, 0.0,0.0).*DFControl.angstrom, DFControl.Point3(1.0,0.0,0.0), projections=[DFControl.Projection(DFControl.orbital(:d), 11, 20)])] 
noncolinmat = convert(DFW.NonColinMatrix, mat, atoms)

@test mat[1] == noncolinmat[1]
@test mat[2, 2] == noncolinmat[2, 2]
@test mat[6, 6] == noncolinmat[11, 11]
@test mat[11, 11] == noncolinmat[6, 6]
@test mat[1, 2] == noncolinmat[1, 2]
@test mat[2, 1] == noncolinmat[2, 1]
@test mat[1, 6] == noncolinmat[1, 11]
@test mat[11, 1] == noncolinmat[6, 1]

@test noncolinmat[atoms[1]] == [noncolinmat[1:5,1:5] noncolinmat[1:5, 11:15];noncolinmat[11:15, 1:5] noncolinmat[11:15, 11:15]]
@test noncolinmat[atoms[1], atoms[2]] == [noncolinmat[1:5,6:10] noncolinmat[1:5, 16:20];noncolinmat[11:15, 6:10] noncolinmat[11:15, 16:20]]

@test noncolinmat[atoms[1], atoms[2], DFW.Up()] == noncolinmat[1:5,6:10]
@test noncolinmat[atoms[1], atoms[2], DFW.Down()] == noncolinmat[11:15, 16:20]

@test noncolinmat[atoms[1], atoms[2], DFW.Up(), DFW.Down()] == noncolinmat[1:5,16:20]
@test noncolinmat[atoms[1], atoms[2], DFW.Down(), DFW.Up()] == noncolinmat[11:15, 6:10]
