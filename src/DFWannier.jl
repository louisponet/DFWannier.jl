#Cleanup don't export everything that doesn't have to be exported
module DFWannier
  # if (Pkg.installed.(["CUDAdrv","CuArrays","CUDAnative"]) .!= [nothing for _=1:3])==[true for __=1:3]
  #   gpu_enabled = false
  # else
  #   gpu_enabled = false
  # end
  using Reexport
  @reexport using DFControl
  using RecipesBase
  using LaTeXStrings
  using StaticArrays
  using GeometryTypes
  # if gpu_enabled
  #   using CuArrays
  #   using CUDAdrv:CuDevice,CuContext,attribute,MAX_THREADS_PER_BLOCK,destroy!
  #   using CUDAnative
  #
  #   dev = CuDevice(0)
  #   ctx = CuContext(dev)
  #   end
  include("types.jl")
  include("typedefs.jl")
  #---#
  export WfcPoint3
  export Wfc3D
  # if gpu_enabled
  #   export Wfc3D_gpu
  # end
  export WannierBand
  include("atom_ext.jl")
  export WanAtom
  include("structure_ext.jl")
  export WanStructure
  export add_wan_data
  export set_soc!

  include("utils.jl")

  include("wan_calcs.jl")
  # if gpu_enabled
  #   include("wan_calcs_gpu.jl")
  # end
  export construct_bloch_sum
  export calc_angmom
  export calc_angmoms
  export calc_angmoms!
  export calc_spins
  export calc_dip
  export calc_dips
  include("hami_calcs.jl")
  include("structure_calcs.jl")
  export calc_observables
  include("plotting.jl")
  include("exchange.jl")
  export AtomOrbInfo
  export exchange_between
  export calculate_exchanges
  include("fileio.jl")
  export read_xsf_file
  export write_xsf_file
  export read_exchanges
  export write_exchanges
  export read_hami_file
end
