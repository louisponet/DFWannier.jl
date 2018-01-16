#Cleanup don't export everything that doesn't have to be exported
module DFWannier
  if (Pkg.installed.(["CUDAdrv","CuArrays","CUDAnative"]) .!= [nothing for _=1:3])==[true for __=1:3]
    gpu_enabled = true
  else
    gpu_enabled = false
  end
  using Reexport
  @reexport using DFControl
  using RecipesBase
  using LaTeXStrings
  using StaticArrays
  if gpu_enabled 
    using CuArrays
    using CUDAdrv:CuDevice,CuContext,attribute,MAX_THREADS_PER_BLOCK,destroy!
    using CUDAnative

    dev = CuDevice(0)
    ctx = CuContext(dev)
  end
  include("types.jl")
  export Atom
  #---#
  export WfcPoint3D
  export Wfc3D
  if gpu_enabled
    export Wfc3D_gpu
  end
  export WannierBand
  export WannierModel

  include("utils.jl") 
  include("fileio.jl")
  export read_xsf_file
  export write_xsf_file
  
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
  include("model_calcs.jl")
  include("plotting.jl")
  include("exchange.jl")
  export AtomOrbInfo
  export WannExchanges
  export exchange_between
  export calculate_exchanges
end