using DFWannier
using Test
using LinearAlgebra
const DFW = DFWannier

assetfile(f) = joinpath("assets", f)


@time @testset "exchanges" begin include("exchanges.jl") end
@time @testset "linalg" begin include("linalg.jl") end
@time @testset "wan_calcs" begin include("wan_calcs.jl") end
