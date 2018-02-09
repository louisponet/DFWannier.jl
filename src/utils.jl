
function sorted_eig(hami)
  # for i=1:size(hami)[1]
  #   hami[i,i] = real(hami[i,i])
  # end
  #
  eigs, eigvec = eig(hami)
  out_eig = similar(eigs)
  out_vec = similar(eigvec)
  perm = sortperm(real(eigs))
  for (i, p) in enumerate(perm)
    out_vec[:,i]=eigvec[:,p]
    out_eig[i] = eigs[p]
  end
  return out_eig, out_vec
end

function get_pauli(T::Type, dir::Symbol, dim::Int)
  out = zeros(Complex{T}, dim, dim)
  if dir == :x
    out[1:div(dim,2),div(dim,2)+1:end]=0.5*eye(T,div(dim,2),div(dim,2))
    out[div(dim,2)+1:end,1:div(dim,2)]=0.5*eye(T,div(dim,2),div(dim,2))
  elseif dir == :y
    out[1:div(dim,2),div(dim,2)+1:end]=-0.5im*eye(T,div(dim,2),div(dim,2))
    out[div(dim,2)+1:end,1:div(dim,2)]=0.5im*eye(T,div(dim,2),div(dim,2))
  else
    out[1:div(dim,2),1:div(dim,2)]=0.5eye(T,div(dim,2),div(dim,2))
    out[div(dim,2)+1:end,div(dim,2)+1:end]=-0.5*eye(T,div(dim,2),div(dim,2))
  end
  return SMatrix{dim, dim, Complex{T}}(out)
end

if gpu_enabled
function host2gpu(wfc::Wfc3D)
  cell=wfc.cell
  return Wfc3D_gpu(CuArray([(p.p.x,p.p.y,p.p.z) for p in wfc.points]),CuArray([p.w for p in wfc.points]),wfc.cell,wfc.atom)
end
function gpu2host(wfc::Wfc3D_gpu)
  return Wfc3D([WfcPoint3D(w,Point3D(p...)) for (w,p) in zip(wfc.values,wfc.grid)],Array(wfc.cell),wfc.atom)
end
end
