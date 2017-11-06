#Cleanup don't export everything that doesn't have to be exported
module DFWannier
  using Reexport
  @reexport using DFControl
  using RecipesBase
  using LaTeXStrings
  include("types.jl")
  export PhysAtom
  #---#
  export WfcPoint3D
  export Wfc3D
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