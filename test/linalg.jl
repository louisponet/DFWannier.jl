for F in (Float32, Float64)
    n = 3
    tup = rand(Complex{F}, n, n)
    tdown = rand(Complex{F}, n, n)
    orig_up = Hermitian((tup + tup')/2)
    orig_dn = Hermitian((tdown + tdown')/2)
    colin = DFW.ColinMatrix(orig_up, orig_dn)
    normal_eig1, normal_eig2 = eigen(orig_up), eigen(orig_dn)
    cache      = DFW.HermitianEigenWs(colin)
    cached_eig = eigen(colin, cache)
    @test sum(normal_eig1.values) + sum(normal_eig2.values) ≈ sum(cached_eig.values)
    @test Array(normal_eig1) ≈ Array(cached_eig)[1:n, 1:n] ≈ DFW.up(colin)
    @test Array(normal_eig2) ≈ Array(cached_eig)[1:n, n+1:end] ≈ DFW.down(colin)
end

for F in (Float32, Float64)
    t = rand(Complex{F}, 50, 50)
    orig       = (t + t')/2
    normal_eig = eigen(orig)
    cache      = DFW.HermitianEigenWs(orig)
    cached_eig = eigen(orig, cache)
    @test sum(normal_eig.values) ≈ sum(cached_eig.values) 
    @test Array(normal_eig) ≈ Array(cached_eig) ≈ orig
end

mat = rand(20, 20)
ats = [DFControl.Atom(name=:Fe, element=DFControl.element(:Fe), position_cart=DFControl.Point3(0.0, 0.0,0.0).*DFControl.angstrom, position_cryst=DFControl.Point3(0.0,0.0,0.0), projections=[DFControl.Projection(Structures.orbital("d"), 1, 10)]), DFControl.Atom(name=:Fe, element=DFControl.element(:Fe), position_cart=DFControl.Point3(1.0, 0.0,0.0).*DFControl.angstrom, position_cryst=DFControl.Point3(1.0,0.0,0.0), projections=[DFControl.Projection(Structures.orbital("d"), 11, 20)])] 
noncolinmat = convert(DFW.NonColinMatrix, mat)

@test mat[1] == noncolinmat[1]
@test mat[2, 2] == noncolinmat[11, 11]
@test mat[6, 6] == noncolinmat[13, 13]
@test mat[11, 11] == noncolinmat[6, 6]
@test mat[1, 2] == noncolinmat[1, 11]
@test mat[2, 1] == noncolinmat[11, 1]
@test mat[1, 6] == noncolinmat[1, 13]
@test mat[11, 1] == noncolinmat[6, 1]

@test noncolinmat[ats[1]] == [noncolinmat[1:5,1:5] noncolinmat[1:5, 11:15];noncolinmat[11:15, 1:5] noncolinmat[11:15, 11:15]]
@test noncolinmat[ats[1], ats[2]] == [noncolinmat[1:5,6:10] noncolinmat[1:5, 16:20];noncolinmat[11:15, 6:10] noncolinmat[11:15, 16:20]]

@test noncolinmat[ats[1], ats[2], DFW.Up()] == noncolinmat[1:5,6:10]
@test noncolinmat[ats[1], ats[2], DFW.Down()] == noncolinmat[11:15, 16:20]

@test noncolinmat[ats[1], ats[2], DFW.Up(), DFW.Down()] == noncolinmat[1:5,16:20]
@test noncolinmat[ats[1], ats[2], DFW.Down(), DFW.Up()] == noncolinmat[11:15, 6:10]
