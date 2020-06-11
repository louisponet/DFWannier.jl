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
noncolinmat = convert(DFW.NonColinMatrix, mat)

@test mat[1] == noncolinmat[1]
@test mat[2, 2] == noncolinmat[11, 11]
@test mat[3, 3] == noncolinmat[2, 2]
@test mat[4, 4] == noncolinmat[12, 12]
@test mat[1, 2] == noncolinmat[1, 11]
@test mat[2, 1] == noncolinmat[11, 1]
@test mat[1, 3] == noncolinmat[1, 2]
@test mat[3, 1] == noncolinmat[2, 1]

proj = DFControl.Projection(DFControl.Orbital(:p, 3, 1), 1, 6)
@test noncolinmat[proj] == [noncolinmat[1:3,1:3] noncolinmat[1:3, 11:13];noncolinmat[11:13, 1:3] noncolinmat[11:13, 11:13]]
proj2 = DFControl.Projection(DFControl.Orbital(:d, 5, 2), 7, 16)
@test noncolinmat[proj, proj2] == [noncolinmat[1:3,4:8] noncolinmat[1:3, 14:18];noncolinmat[11:13, 4:8] noncolinmat[11:13, 14:18]]

@test noncolinmat[proj, proj2, DFW.Up()] == noncolinmat[1:3,4:8]
@test noncolinmat[proj, proj2, DFW.Down()] == noncolinmat[11:13, 14:18]

@test noncolinmat[proj, proj2, DFW.Up(), DFW.Down()] == noncolinmat[1:3,14:18]
@test noncolinmat[proj, proj2, DFW.Down(), DFW.Up()] == noncolinmat[11:13, 4:8]
