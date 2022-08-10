#Cleanup don't export everything that doesn't have to be exported
module DFWannier
using Reexport

@reexport using DFControl
using RecipesBase
using LaTeXStrings
using LinearAlgebra
using Base.Threads
using FortranFiles
using DelimitedFiles
using Unitful
using ProgressMeter
using Requires
import Base: @propagate_inbounds
using FastLapackInterface
import LinearAlgebra.BLAS: libblas
import LinearAlgebra: eigen, eigen!

const DFW = DFWannier
export DFW

include("utils.jl")
include("tight_binding.jl")
include("reciprocal.jl")
include("magnetic.jl")

include("wannierfunction.jl")
include("linalg.jl")

include("plotting.jl")
include("exchange.jl")
include("fileio.jl")
include("berry.jl")
export HamiltonianKGrid

export wannierbands, readhami, calc_exchanges, read_colin_hami, 
       kpdos, energy_bins
export uniform_kgrid
export Up, Down, WannierFunction, calc_greens_functions, ExchangeKGrid, HamiltonianKGrid
export generate_wannierfunctions
export wan_hash
export write_xsf, read_chk, read_spn, S_R
function __init__()
    @require Glimpse = "f6e19d58-12a4-5927-8606-ac30a9ce9b69" include("glimpse.jl")
end
end
