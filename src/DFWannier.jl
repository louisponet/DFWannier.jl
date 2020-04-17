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
	using DelimitedFiles
	using Unitful
	using ProgressMeter
	using UnsafeArrays

	include("linalg.jl")
	include("types.jl")
	include("wan_calcs.jl")
	include("k_grid.jl")
	include("hami_calcs.jl")
	# include("structure_calcs.jl")
	include("plotting.jl")
	include("exchange.jl")
	include("fileio.jl")
	include("berry.jl")
	# include("orbital_angular_momentum.jl")
    export HamiltonianKGrid
	export wannierbands, readhami, calc_exchanges, read_colin_hami, character_contributions, kpdos
	export uniform_kgrid
end
