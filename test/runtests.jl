using DFWannier
using Test
using LinearAlgebra
const DFW = DFWannier

assetfile(f) = joinpath("assets", f)


@time @testset "linalg" begin include("linalg.jl") end
@time @testset "hami_calcs" begin include("hami_calcs.jl") end
# @time @testset "berry" begin include("berry.jl") end
@time @testset "exchanges" begin include("exchanges.jl") end
@time @testset "wan_calcs" begin include("wan_calcs.jl") end
