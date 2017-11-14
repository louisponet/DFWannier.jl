# EXCLUDE FROM TESTING
# this file doesn't have an entry point, see `verify.jl` instead

# Fast parallel reduction for Kepler hardware
# - uses shuffle and shared memory to reduce efficiently
# - support for large arrays
#
# Based on devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

using CUDAdrv, CUDAnative


#
# Main implementation
#

# Reduce a value across a warp
@inline function reduce_warp(op::Function, val::T)::T where {T}
  offset = CUDAnative.warpsize() ÷ UInt32(2)
  # TODO: this can be unrolled if warpsize is known...
  while offset > 0
    val = op(val, shfl_down(val, offset))
    offset ÷= UInt32(2)
  end
  return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::Function, val_x::T,val_y::T,val_z::T)::T where {T}
  # shared mem for 32 partial sums
  shared = @cuStaticSharedMem(T, 32)
  
  # TODO: use fldmod1 (JuliaGPU/CUDAnative.jl#28)
  wid_x  = div(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  wid_y  = div(threadIdx().y-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  wid_z  = div(threadIdx().z-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  lane_x = rem(threadIdx().x-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  lane_y = rem(threadIdx().y-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  lane_z = rem(threadIdx().z-UInt32(1), CUDAnative.warpsize()) + UInt32(1)
  
  # each warp performs partial reduction
  val_x = reduce_warp(op, val_x)
  val_y = reduce_warp(op, val_y)
  val_z = reduce_warp(op, val_z)
  
  # write reduced value to shared memory
  if lane_x == 1
    @inbounds shared[wid_x] = val_x
  end
  if lane_y == 1
    @inbounds shared[wid_y] = val_y
  end
  if lane_z == 1
    @inbounds shared[wid_z] = val_z
  end 
  # wait for all partial reductions
  sync_threads()
  
  # read from shared memory only if that warp existed
  @inbounds val_x = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)
  @inbounds val_y = (threadIdx().y <= fld(blockDim().y, CUDAnative.warpsize())) ? shared[lane] : zero(T)
  @inbounds val_z = (threadIdx().z <= fld(blockDim().z, CUDAnative.warpsize())) ? shared[lane] : zero(T)
  
  # final reduce within first warp
  if wid_x == 1
    val_x = reduce_warp(op, val_x)
  end
  if wid_y == 1
    val_y = reduce_warp(op, val_x)
  end
  if wid_z == 1
    val_z = reduce_warp(op, val_x)
  end
  
  return val_x+valy_y+val_z
end

# Reduce an array across a complete grid
function reduce_grid(op::Function, input::CuDeviceVector{T}, output::CuDeviceVector{T},
  len::CuDeviceVector{Integer}) where {T}
  
  # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
  val = zero(T)
  
  # reduce multiple elements per thread (grid-stride loop)
  # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
  i_x = (blockIdx().x-UInt32(1)) * blockDim().x + threadIdx().x
  i_y = (blockIdx().y-UInt32(1)) * blockDim().y + threadIdx().y
  i_z = (blockIdx().z-UInt32(1)) * blockDim().z + threadIdx().z
  step_x = blockDim().x * gridDim().x
  step_y = blockDim().y * gridDim().y
  step_z = blockDim().z * gridDim().z
  while i_x <= len[1]
    while i_y <= len[2]
      while i_z <= len[3]
        @inbounds val = op(val, input[i])
        i_z+=step_z
      end
      i_y += step_y
    end
    i_x += step_x
  end
  
  val = reduce_block(op, val)
  
  if threadIdx().x == UInt32(1)
    @inbounds output[blockIdx().x] = val
  end
  
  return
end

"""
Reduce a large array.
Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce(op::Function, input::CuVector{T}, output::CuVector{T}) where {T}
  len = length(input)
  
  # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
  threads = 512
  blocks = min((len + threads - 1) ÷ threads, 1024)
  
  # the output array must have a size equal to or larger than the number of thread blocks
  # in the grid because each block writes to a unique location within the array.
  if length(output) < blocks
    throw(ArgumentError("output array too small, should be at least $blocks elements"))
  end
  
  @cuda (blocks,threads) reduce_grid(op, input, output, Int32(len))
  @cuda (1,1024) reduce_grid(op, output, output, Int32(blocks))
  
  return
end
ctx = CuCurrentContext()
dev = device(ctx)
if capability(dev) < v"3.0"
  warn("this example requires a newer GPU")
  exit(0)
end

len = 10^7
input = ones(Int32, len)

# CPU
cpu_val = reduce(+, input)

# CUDAnative
let
  gpu_input = CuArray(input)
  gpu_output = similar(gpu_input)
  gpu_reduce(+, gpu_input, gpu_output)
  gpu_val = Array(gpu_output)[1]
  @assert cpu_val == gpu_val
end



