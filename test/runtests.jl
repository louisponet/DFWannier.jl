using DFWannier
using Test
using LinearAlgebra
const DFW = DFWannier
@time @testset "wan_calcs" begin include("wan_calcs.jl") end
@time @testset "linalg" begin include("linalg.jl") end
