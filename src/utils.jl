div1(x, y) = div(x - 1, y) + 1
function sorted_eig(hami)
    eigs, eigvec = (eigen(hami)...,)
    out_eig = similar(eigs)
    out_vec = similar(eigvec)
    perm = sortperm(real(eigs))
    for (i, p) in enumerate(perm)
    out_vec[:,i]=eigvec[:,p]
    out_eig[i] = eigs[p]
    end
    return out_eig, out_vec
end

function pauli(T::Type, dir::Symbol, dim::Int)
  out = zeros(Complex{T}, dim, dim)
  if dir == :x
    out[1:div(dim,2),div(dim,2)+1:end]=0.5*Matrix{T}(I,div(dim,2),div(dim,2))
    out[div(dim,2)+1:end,1:div(dim,2)]=0.5*Matrix{T}(I,div(dim,2),div(dim,2))
  elseif dir == :y
    out[1:div(dim,2),div(dim,2)+1:end]=-0.5im*Matrix{T}(I,div(dim,2),div(dim,2))
    out[div(dim,2)+1:end,1:div(dim,2)]=0.5im*Matrix{T}(I,div(dim,2),div(dim,2))
  else
    out[1:div(dim,2),1:div(dim,2)]=0.5Matrix{T}(I,div(dim,2),div(dim,2))
    out[div(dim,2)+1:end,div(dim,2)+1:end]=-0.5*Matrix{T}(I,div(dim,2),div(dim,2))
  end
  return out
end

@inline randpolϕ(::Type{T}=Float64) where T = 2π * rand(T)
@inline randpolθ(::Type{T}=Float64) where T = acos(1 - 2rand(T))

@inline randpolθϕ(::Type{T}=Float64) where T = (randpolθ(T), randpolϕ(T))
@inline randθsϕs(::Type{T}, n) where T = [[randpolθ(T) for i=1:n] ; [randpolϕ(T) for j=1:n]]
@inline randθsϕs(n) = randθsϕs(Float64, n)

@inline polar2xyz(r, θ, ϕ) = Vec3(r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ))
# if gpu_enabled
# function host2gpu(wfc::Wfc3D)
#   cell=wfc.cell
#   return Wfc3D_gpu(CuArray([(p.p.x,p.p.y,p.p.z) for p in wfc.points]),CuArray([p.w for p in wfc.points]),wfc.cell,wfc.atom)
# end
# function gpu2host(wfc::Wfc3D_gpu)
#   return Wfc3D([WfcPoint3(w,Point3(p...)) for (w,p) in zip(wfc.values,wfc.grid)],Array(wfc.cell),wfc.atom)
# end
# end
