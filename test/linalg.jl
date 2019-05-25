for F in (Float32, Float64)
	t = rand(Complex{F}, 50, 50)
	orig       = (t + t')/2
	normal_eig = eigen(orig)
	cache      = DFW.EigCache(orig)
	cached_eig = eigen(orig, cache)
	@test sum(normal_eig.values) ≈ sum(cached_eig.values) 
	@test Array(normal_eig) ≈ Array(cached_eig) ≈ orig

	tvals = zeros(F, 50)
	torig = copy(orig)
	@test Array(eigen!(tvals, torig, cache)) ≈ orig
end

