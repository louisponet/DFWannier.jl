#Cleanup don't export everything that doesn't have to be exported
module DFWannier
  if Pkg.installed.(["CUDAdrv","CuArrays","CUDAnative"]) != [nothing for _=1:3]
    gpu_enabled = true
  else
    gpu_enabled = false
  end
  using Reexport
  @reexport using DFControl
  using RecipesBase
  using LaTeXStrings
  if gpu_enabled 
    using CuArrays
    using CUDAdrv:CuDevice,CuContext,attribute,MAX_THREADS_PER_BLOCK,destroy!
    using CUDAnative

    dev = CuDevice(0)
    ctx = CuContext(dev)
  end
  include("types.jl")
  export PhysAtom
  #---#
  export WfcPoint3D
  export Wfc3D
  if gpu_enabled
    export Wfc3D_gpu
  end
  export WannierBand
  export WannierModel

  include("utils.jl") 
  include("file_processing.jl")
  export read_xsf_file
  export write_xsf_file

  include("wannier_calcs.jl")
  export construct_bloch_sum
  export calculate_angmom
  export calculate_angmoms
  export calculate_spins
  export calculate_dip
  export calculate_dips

  include("hami_calcs.jl")
  include("model_calcs.jl")
  include("plotting.jl")
end