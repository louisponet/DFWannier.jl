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
