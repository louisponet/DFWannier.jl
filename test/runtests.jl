using DFWannier
using Test
using LinearAlgebra
const DFW = DFWannier

assetfile(f) = joinpath("assets", f)
@time @testset "hami_calcs" begin include("hami_calcs.jl") end
@time @testset "exchanges" begin include("exchanges.jl") end
#@time @testset "linalg" begin include("linalg.jl") endexchanges
#@time @testset "wan_calcs" begin include("wan_calcs.jl") end
