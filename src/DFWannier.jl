#Cleanup don't export everything that doesn't have to be exported
module DFWannier
	using Reexport

	@reexport using DFControl
	const DFC = DFControl

	using StaticArrays 
	using RecipesBase
	using LaTeXStrings
	using InplaceOps
	using LinearAlgebra
	using Base.Threads
	using Documenter
	using Parameters
	using StructArrays
	using FortranFiles

	include("linalg.jl")
	include("types.jl")
	include("wan_calcs.jl")
	include("hami_calcs.jl")
	include("structure_calcs.jl")
	include("plotting.jl")
	include("exchange.jl")
	include("fileio.jl")

	export wannierbands, readhami, calc_exchanges, read_colin_hami
end
