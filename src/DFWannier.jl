#Cleanup don't export everything that doesn't have to be exported
module DFWannier
  if (Pkg.installed.(["CUDAdrv","CuArrays","CUDAnative"]) .!= [nothing for _=1:3])==[true for __=1:3]
    gpu_enabled = false
  else
    gpu_enabled = false
  end
  using Reexport
  @reexport using DFControl
  using RecipesBase
  using LaTeXStrings
  using StaticArrays
  using GeometryTypes
  if gpu_enabled
    using CuArrays
    using CUDAdrv:CuDevice,CuContext,attribute,MAX_THREADS_PER_BLOCK,destroy!
    using CUDAnative

    dev = CuDevice(0)
    ctx = CuContext(dev)
    end
  include("types.jl")
  include("typedefs.jl")
  #---#
  export WfcPoint3D
  export Wfc3D
  if gpu_enabled
    export Wfc3D_gpu
  end
  export WannierBand
  export WannierModel
  include("atom_ext.jl")
  include("structure_ext.jl")
  export add_wan_data
  export set_soc!

  include("utils.jl")

  include("wan_calcs.jl")
  if gpu_enabled
    include("wan_calcs_gpu.jl")
  end
  export construct_bloch_sum
  export calculate_angmom
  export calculate_angmoms
  export calculate_spins
  export calculate_dip
  export calculate_dips

  include("hami_calcs.jl")
  include("structure_calcs.jl")
  # include("model_calcs.jl")
  include("plotting.jl")
  include("exchange.jl")
  export AtomOrbInfo
  export WannExchanges
  export exchange_between
  export calculate_exchanges
  include("fileio.jl")
  export read_xsf_file
  export write_xsf_file
  export read_exchanges
  export write_exchanges
end
