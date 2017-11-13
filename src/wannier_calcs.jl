
"Constructs the bloch sum at k for the wavefunction supplied"
function construct_bloch_sum(wfc::Wfc3D{T}, k::Array) where T<:AbstractFloat
  points = zeros(Complex{T},size(wfc.points))
  dim_a = size(points)[1]
  dim_b = size(points)[2]
  dim_c = size(points)[3]
  R::Point3D{T} = Point3D(0.0)
  for R1=-1:1,R2=-1:1,R3=-1:1
    R= R1*wfc.cell[1]+R2*wfc.cell[2]+R3*wfc.cell[3]
    c = exp(dot(-2*pi*k,[R1,R2,R3])*1im)
    ind1,ind2 = find_start(wfc,R,27)
    i3 = ind1[3]
    j3 = ind2[3]
    while i3 <= dim_c && j3 <=dim_c
      i2 = ind1[2]
      j2 = ind2[2]
      while i2 <= dim_b && j2 <= dim_b
        i1 = ind1[1]
        j1 = ind2[1]
        while i1 <=dim_a && j1 <= dim_a
          points[i1,i2,i3]+=c*wfc.points[j1,j2,j3].w
          i1+=1
          j1+=1
        end
        i2+=1
        j2+=1
      end
      i3+=1
      j3+=1
    end
  end
  tmp_points=Array{WfcPoint3D{T},3}(dim_a,dim_b,dim_c)
  for i2=1:dim_c
    for i1=1:dim_b
      for i=1:dim_a
        tmp_points[i,i1,i2] = WfcPoint3D(points[i,i1,i2],wfc.points[i,i1,i2].p)
      end
    end
  end
  return Wfc3D(tmp_points,wfc.cell,wfc.atom)
end

function find_start(wfc,R,partitions)::Tuple{Tuple{Int64,Int64,Int64},Tuple{Int64,Int64,Int64}}
  part_1D = partitions^(1/3)
  # part_1D = partitions
  dim_a = size(wfc.points)[1]
  dim_b = size(wfc.points)[2]
  dim_c = size(wfc.points)[3]
  stride_a::Int64 = dim_a/part_1D
  stride_b::Int64 = dim_b/part_1D
  stride_c::Int64 = dim_c/part_1D
  anchors = [wfc.points[a,b,c].p for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
  shifted_anchors = [wfc.points[a,b,c].p-R for a=1:stride_a:dim_a,b=1:stride_a:dim_b,c=1:stride_c:dim_c]
  for i in eachindex(anchors)
    for j in eachindex(shifted_anchors)
      if norm(anchors[i]-shifted_anchors[j])<0.00001
        tmp1 = ind2sub(anchors,i)
        tmp2 = ind2sub(shifted_anchors,j)
        ind1 = ((tmp1[1]-1)*stride_a+1,(tmp1[2]-1)*stride_b+1,(tmp1[3]-1)*stride_c+1)
        ind2 = ((tmp2[1]-1)*stride_a+1,(tmp2[2]-1)*stride_b+1,(tmp2[3]-1)*stride_c+1)
        return ind1,ind2
      end
    end
  end
end



"Calculates the angular momentum between two wavefunctions and around the center."
function calculate_angmom(wfc1::Wfc3D{T}, wfc2::Wfc3D{T}, center::Point3D{T}) where T<:AbstractFloat
  origin = wfc1[1,1,1].p
  a = wfc1[2,1,1].p - origin
  b = wfc1[1,2,1].p - origin
  c = wfc1[1,1,2].p - origin
  V = inv([convert(Array,a) convert(Array,b) convert(Array,c)])
  center_x = center.x
  center_y = center.y
  center_z = center.z
  dadx = V[1,1]
  dbdx = V[2,1]
  dcdx = V[3,1]
  dady = V[1,2]
  dbdy = V[2,2]
  dcdy = V[3,2]
  dadz = V[1,3]
  dbdz = V[2,3]
  dcdz = V[3,3]
  Lx = zero(Complex{T})
  Ly = zero(Complex{T})
  Lz = zero(Complex{T})
  n1 = zero(T)
  n2 = zero(T)
  @inbounds for i2 = 2:size(wfc1.points)[3]-1
    @inbounds for i1 = 2:size(wfc1.points)[2]-1
      @inbounds for i = 2:size(wfc1.points)[1]-1
        
        ddax = (wfc2[i+1,i1,i2].w-wfc2[i-1,i1,i2].w)*dadx
        ddbx = (wfc2[i,i1+1,i2].w-wfc2[i,i1-1,i2].w)*dbdx
        ddcx = (wfc2[i,i1,i2+1].w-wfc2[i,i1,i2-1].w)*dcdx
        ddx = ddax+ddbx+ddcx
        
        dday = (wfc2[i+1,i1,i2].w-wfc2[i-1,i1,i2].w)*dady
        ddby = (wfc2[i,i1+1,i2].w-wfc2[i,i1-1,i2].w)*dbdy
        ddcy = (wfc2[i,i1,i2+1].w-wfc2[i,i1,i2-1].w)*dcdy
        ddy = dday+ddby+ddcy
        
        ddaz = (wfc2[i+1,i1,i2].w-wfc2[i-1,i1,i2].w)*dadz
        ddbz = (wfc2[i,i1+1,i2].w-wfc2[i,i1-1,i2].w)*dbdz
        ddcz = (wfc2[i,i1,i2+1].w-wfc2[i,i1,i2-1].w)*dcdz
        ddz = ddaz+ddbz+ddcz
        
        Lx += conj(wfc1[i,i1,i2].w)*-1im*((wfc2[i,i1,i2].p.y-center_y)*ddz-(wfc2[i,i1,i2].p.z-center_z)*ddy)
        Ly += conj(wfc1[i,i1,i2].w)*-1im*((wfc2[i,i1,i2].p.z-center_z)*ddx-(wfc2[i,i1,i2].p.x-center_x)*ddz)
        Lz += conj(wfc1[i,i1,i2].w)*-1im*((wfc2[i,i1,i2].p.x-center_x)*ddy-(wfc2[i,i1,i2].p.y-center_y)*ddx)
        n1+= norm(wfc1[i,i1,i2].w)^2
        n2+= norm(wfc2[i,i1,i2].w)^2
      end
    end
  end
  n=sqrt(n1*n2)
  # return Lx,Ly,Lz
  return Lx/n,Ly/n,Lz/n
end



"Calculates the angular momenta between two wavefunctions, around the atom of the second wavefunction."
function calculate_angmom(wfc1::Wfc3D{T},wfc2::Wfc3D{T}) where T<:AbstractFloat
  if wfc1.atom != wfc2.atom
    return zero(Complex{T}),zero(Complex{T}),zero(Complex{T})
  else
    return calculate_angmom(wfc1,wfc2,wfc2.atom.center)
  end
end

"Calculates the angular momenta between all the supplied wavefunctions"
function calculate_angmoms(wfcs::Array{Wfc3D{T}}) where T<:AbstractFloat
  out = Array{Array{Complex{T},1},2}((size(wfcs)[1],size(wfcs)[1]))
  for (i,wfc1) in enumerate(wfcs)
    for (i1,wfc2) in enumerate(wfcs)
      out[i,i1]=[calculate_angmom(wfc1,wfc2)...]
    end
  end
  return [out fill([zero(Complex{T}) for i=1:3],size(out));fill([zero(Complex{T}) for i=1:3],size(out)) out]
end

"Calculates the spins between the supplied wavefunctions"
function calculate_spins(wfcs::Array{Wfc3D{T}}) where T<:AbstractFloat
  dim = length(wfcs)
  dim_2 = div(dim,2)
  s_x = get_pauli(T,:x,2*dim)
  s_y = get_pauli(T,:y,2*dim)
  s_z = get_pauli(T,:z,2*dim)
  return s_x,s_y,s_z
end

"Calculates the dipole term between two wavefunctions"
function calculate_dip(wfc1::Wfc3D{T},wfc2::Wfc3D{T}) where T<:AbstractFloat
  out_x = zero(Complex{T})
  out_y = zero(Complex{T})
  out_z = zero(Complex{T})
  n1 = zero(Complex{T})
  n2 = zero(Complex{T})
  for (p1,p2) in zip(wfc1.points,wfc2.points)
    out_x += conj(p1.w)*p2.w*p1.p.x
    out_y += conj(p1.w)*p2.w*p1.p.y
    out_z += conj(p1.w)*p2.w*p1.p.z
    n1 += norm(p1.w)^2
    n2 += norm(p2.w)^2
  end
  n = sqrt(n1*n2)
  return Point3D(real(out_x/n),real(out_y/n),real(out_z/n))
end

"Calculates all dipole terms between the wavefunctions."
function calculate_dips(wfcs::Array{<:Wfc3D})
  out = Array{Point3D,2}((size(wfcs)[1],size(wfcs)[1]))
  for (i,wfc1) in enumerate(wfcs)
    for (i1,wfc2) in enumerate(wfcs)
      out[i,i1]=calculate_cm(wfc1,wfc2)
    end
  end
  return out
end

"Calculates the dipoles from the supplied wannier dipole output."
function calculate_k_dips(dip_raw::Array{Tuple{Int,Int,Int,Int,Int,Point3D{T}}},k_points::Array) where T<:AbstractFloat
  dim = 0
  for i=1:length(dip_raw)
    d = dip_raw[i][4]
    if d>dim
      dim = d
    else
      break
    end
  end
  out = zeros(Point3D{T},(dim,dim))
  tmp = [[zero(Complex{T}),zero(Complex{T}),zero(Complex{T})] for i=1:dim,i1=1:dim]
  for i=1:size(dip_raw)[1]
    d = dip_raw[i]
    complex_part = 2*pi*(k_points[1]*d[1]+k_points[2]*d[2]+k_points[3]*d[3])
    tmp[d[4],d[5]][1] += d[6].x*exp(-1im*complex_part)
    tmp[d[4],d[5]][2] += d[6].y*exp(-1im*complex_part)
    tmp[d[4],d[5]][3] += d[6].z*exp(-1im*complex_part)
  end
  for i in eachindex(out)
    out[i]=Point3D(real(tmp[i][1]),real(tmp[i][2]),real(tmp[i][3]))
  end
  return [out zeros(out);zeros(out) out]
end

#------------------------Not currently used beyond here---------------------------------#
function calculate_overlap_angmom(wfc1::Wfc3D{T},wfc2::Wfc3D{T},n_overlaps::Int64) where T<:AbstractFloat
  if wfc1.atom!=wfc2.atom
    return [(zero(Complex{T}),zero(Complex{T}),zero(Complex{T})) for i=1:2*n_overlaps+1,i1=1:2*n_overlaps+1,i2=1:2*n_overlaps+1]
  end
  out_moms = Array{Tuple{Complex{T},Complex{T},Complex{T}},3}(2*n_overlaps+1,2*n_overlaps+1,2*n_overlaps+1)
  dim_a = size(wfc1.points)[1]
  dim_b = size(wfc1.points)[2]
  dim_c = size(wfc1.points)[3]
  origin = wfc1[1,1,1].p
  a = wfc1[2,1,1].p - origin
  b = wfc1[1,2,1].p - origin
  c = wfc1[1,1,2].p - origin
  V = inv([convert(Array,a) convert(Array,b) convert(Array,c)])
  center_x = wfc1.atom.center.x
  center_y = wfc1.atom.center.y
  center_z = wfc1.atom.center.z
  dadx = V[1,1]
  dbdx = V[2,1]
  dcdx = V[3,1]
  dady = V[1,2]
  dbdy = V[2,2]
  dcdy = V[3,2]
  dadz = V[1,3]
  dbdz = V[2,3]
  dcdz = V[3,3]
  for R1=-n_overlaps:n_overlaps,R2=-n_overlaps:n_overlaps,R3=-n_overlaps:n_overlaps
    R = R1*wfc1.cell[1]+R2*wfc1.cell[2]+R3*wfc1.cell[3]
    Lx = zero(Complex{T})
    Ly = zero(Complex{T})
    Lz = zero(Complex{T})
    # n1 = zero(T)
    # n2 = zero(T)
    ind1,ind2 = find_start(wfc1,R,n_overlaps+1)
    i2 = ind1[3]+1
    j2 = ind2[3]+1
    while i2 < dim_c && j2 <dim_c
      i1 = ind1[2]+1
      j1 = ind2[2]+1
      while i1 < dim_b && j1 < dim_b
        i = ind1[1]+1
        j = ind2[1]+1
        while i <dim_a && j < dim_a
          ddax = (wfc2[j+1,j1,j2].w-wfc2[j-1,j1,j2].w)*dadx
          ddbx = (wfc2[j,j1+1,j2].w-wfc2[j,j1-1,j2].w)*dbdx
          ddcx = (wfc2[j,j1,j2+1].w-wfc2[j,j1,j2-1].w)*dcdx
          ddx = ddax+ddbx+ddcx
          
          dday = (wfc2[j+1,j1,j2].w-wfc2[j-1,j1,j2].w)*dady
          ddby = (wfc2[j,j1+1,j2].w-wfc2[j,j1-1,j2].w)*dbdy
          ddcy = (wfc2[j,j1,j2+1].w-wfc2[j,j1,j2-1].w)*dcdy
          ddy = dday+ddby+ddcy
          
          ddaz = (wfc2[j+1,j1,j2].w-wfc2[j-1,j1,j2].w)*dadz
          ddbz = (wfc2[j,j1+1,j2].w-wfc2[j,j1-1,j2].w)*dbdz
          ddcz = (wfc2[j,j1,j2+1].w-wfc2[j,j1,j2-1].w)*dcdz
          ddz = ddaz+ddbz+ddcz
          Lx += conj(wfc1[i,i1,i2].w)*-1im*((wfc1[i,i1,i2].p.y-center_y)*ddz-(wfc1[i,i1,i2].p.z-center_z)*ddy)
          Ly += conj(wfc1[i,i1,i2].w)*-1im*((wfc1[i,i1,i2].p.z-center_z)*ddx-(wfc1[i,i1,i2].p.x-center_x)*ddz)
          Lz += conj(wfc1[i,i1,i2].w)*-1im*((wfc1[i,i1,i2].p.x-center_x)*ddy-(wfc1[i,i1,i2].p.y-center_y)*ddx)
          # n1 += norm(wfc1[i,i1,i2].w)^2
          # n2 += norm(wfc2[i,i1,i2].w)^2
          i+=1
          j+=1
        end
        i1+=1
        j1+=1
      end
      i2+=1
      j2+=1
    end
    out_moms[R1+n_overlaps+1,R2+n_overlaps+1,R3+n_overlaps+1] = (Lx,Ly,Lz)
  end
  n1 = calculate_normalization_constant(wfc1)
  n2 = calculate_normalization_constant(wfc2)
  n = sqrt(n1*n2)
  for i in eachindex(out_moms)
    out_moms[i]=(out_moms[i][1]/n,out_moms[i][2]/n,out_moms[i][3]/n)
  end
  return out_moms
end

function calculate_overlap_angmoms(wfcs::Array{Wfc3D{T}},n_overlaps::Int64) where T<:AbstractFloat
  out = Array(Tuple{Complex{T},Complex{T},Complex{T}},length(wfcs),length(wfcs),n_overlaps*2+1,n_overlaps*2+1,n_overlaps*2+1)
  for i1=1:length(wfcs)
    for i=1:length(wfcs)
      out[i,i1,:,:,:] = calculate_overlap_angmom(wfcs[i],wfcs[i1],n_overlaps)
    end
  end
  return out
end

#this is for the bloch sum approach doesnt work
function calculate_angmoms(angmoms::Array{Tuple{Complex{T},Complex{T},Complex{T}},5},k::Array{T,1}) where T<:AbstractFloat
  dim_wfcs = size(angmoms)[1]
  out = [[zero(Complex{T}) for i=1:3] for i1 = 1:dim_wfcs,i2=1:dim_wfcs]
  R1 = -div(size(angmoms)[3],2):div(size(angmoms)[3],2)
  R2 = -div(size(angmoms)[4],2):div(size(angmoms)[4],2)
  R3 = -div(size(angmoms)[5],2):div(size(angmoms)[5],2)
  for i6 = 1:size(angmoms)[5]
    for i5 = 1:size(angmoms)[4]
      for i4 = 1:size(angmoms)[3]
        for i3=1:size(angmoms)[5]
          R3_t = R3[i6]+R3[i3]
          if R3_t < R3[1] | R3_t>R3[end]
            continue
          end
          for i2=1:size(angmoms)[4]
            R2_t = R2[i5]+R2[i2]
            if R2_t < R2[1] | R2_t>R2[end]
              continue
            end
            for i1=1:size(angmoms)[3]
              R1_t = R3[i4]+R3[i1]
              if R1_t < R1[1] | R1_t>R1[end]
                continue
              end
              for n2=1:dim_wfcs
                for n1=1:dim_wfcs
                  Lx_t,Ly_t,Lz_t = angmoms[n1,n2,i1,i2,i3][1],angmoms[n1,n2,i1,i2,i3][2],angmoms[n1,n2,i1,i2,i3][3]
                  c = exp(dot(-2*pi*k,[R1_t,R2_t,R3_t])*1im)
                  out[n1,n2] += [c*Lx_t,c*Ly_t,c*Lz_t]
                end
              end
            end
          end
        end
      end
    end
  end
  tmp = [[zero(Complex{T}) for i=1:3] for i1=1:size(angmoms)[1],i2=1:size(angmoms)[2]]
  out1 = [[out[1:4,1:4] tmp[1:4,1:4];tmp[1:4,1:4] tmp[1:4,1:4]] tmp;tmp [out[1:4,1:4] tmp[1:4,1:4];tmp[1:4,1:4] tmp[1:4,1:4]]]
  out2 = [[tmp[1:4,1:4] tmp[1:4,1:4];tmp[1:4,1:4] out[5:8,5:8]] tmp;tmp [tmp[1:4,1:4] tmp[1:4,1:4];tmp[1:4,1:4] out[5:8,5:8]]]
  return out1,out2
end

function add_overlap_cm!(overlaps::Wfc3D{T},wfc1::Wfc3D{T},wfc2::Wfc3D{T},prefac::T,R::Point3D{T}) where T
  dim_a,dim_b,dim_c = size(wfc1.points)
  n1 = zero(Complex{T})
  n2 = zero(Complex{T})
  ind1,ind2 = find_start(wfc1,R,27)
  i3 = ind1[3]
  j3 = ind2[3]
  while i3<=dim_c && j3 <=dim_c
    i2 = ind1[2]
    j2 = ind2[2]
    while i2<=dim_b && j2 <=dim_b
      i1 = ind1[1]
      j1 = ind2[1]
      while i1<=dim_a && j1 <=dim_a
        w1 = wfc1.points[i1,i2,i3].w
        w2 = wfc2.points[j1,j2,j3].w
        tmp = overlaps[i1,i2,i3].w
        overlaps.points[i1,i2,i3] = WfcPoint3D{T}(prefac*real(conj(w1)*w2)*wfc1.points[i1,i2,i3].p.z+tmp,wfc1.points[i1,i2,i3].p)
        i1+=1
        j1+=1
      end
      i2+=1
      j2+=1
    end
    i3+=1
    j3+=1
  end
end

function add_distribution!(distribution::Wfc3D{T},wfc1::Wfc3D{T},R::Point3D{T}) where T
  dim_a,dim_b,dim_c = size(wfc1.points)
  
  ind1,ind2 = find_start(wfc1,R,27)
  i3 = ind1[3]
  j3 = ind2[3]
  while i3<=dim_c && j3 <=dim_c
    i2 = ind1[2]
    j2 = ind2[2]
    while i2<=dim_b && j2 <=dim_b
      i1 = ind1[1]
      j1 = ind2[1]
      while i1<=dim_a && j1 <=dim_a
        w = wfc1.points[j1,j2,j3].w
        tmp = distribution.points[i1,i2,i3].w
        distribution.points[i1,i2,i3] = WfcPoint3D{T}(tmp+w,wfc1.points[i1,i2,i3].p)
        i1+=1
        j1+=1
      end
      i2+=1
      j2+=1
    end
    i3+=1
    j3+=1
  end
end

function calculate_pot(wfc1::Wfc3D{T},wfc2::Wfc3D{T},potential::Array{T,3}) where T<:AbstractFloat
  dim_a,dim_b,dim_c = (size(wfc1.points)...)
  i1s::Int64,i2s::Int64,i3s::Int64 = 29,29,29
  i1m::Int64,i2m::Int64,i3m::Int64 = dim_a*2/3+1,dim_b*2/3+1,dim_c*2/3+1
  out = zero(Complex{T})
  n1 = zero(Complex{T})
  n2 = zero(Complex{T})
  for i3=i3s:i3m-1
    for i2=i2s:i2m-1
      for i1=i1s:i1m-1
        p1 = wfc1.points[i1,i2,i3]
        p2 = wfc2.points[i1,i2,i3]
        out += conj(p1.w)*p2.w*potential[i1-i1s+1,i2-i2s+1,i3-i3s+1]
        n1 += norm(p1.w)^2
        n2 += norm(p2.w)^2
      end
    end
  end
  n = sqrt(n1*n2)
  return out/n
end

function calculate_tmp_pot(k_wfcs,potential)
  out = Array{Complex,2}((size(k_wfcs)[1],size(k_wfcs)[1]))
  for (i,wfc1) in enumerate(k_wfcs)
    for (i1,wfc2) in enumerate(k_wfcs)
      out[i,i1]= calculate_pot(wfc1,wfc2,potential)
    end
  end
  return [out zeros(out);zeros(out) out]
end

function calculate_dip_mesh_soc(model::WannierModel{T},k_point,band) where T
  points = similar(model.wfcs[1].points,Tuple{Point3D{T},Point3D{T}})
  k_wfcs = Array{Wfc3D{T},1}(size(model.wfcs)[1])
  for (i,wfc) in enumerate(model.wfcs)
    k_wfcs[i] = construct_bloch_sum(wfc,k_point)
  end
  dim_a = size(points)[1]
  dim_b = size(points)[2]
  dim_c = size(points)[3]
  for wfc in k_wfcs
    for ic=1:dim_c
      for ib=1:dim_b
        for ia=1:dim_a
          if ia <= dim_a/7 || ib <= dim_b/7 || ic <= dim_c/7
            wfc.points[ia,ib,ic] = WfcPoint3D(0.0im,wfc.points[ia,ib,ic].p)
          elseif ia>= dim_a*(1-1/7) || ib >= dim_b*(1-1/7) || ic >= dim_c*(1-1/7)
            wfc.points[ia,ib,ic] = WfcPoint3D(0.0im,wfc.points[ia,ib,ic].p)
          else
            continue
          end
        end
      end
    end
  end
  hami = construct_SOC_hami(hami_from_k(model.hami_raw,k_point),model.wfcs)
  eigval,eigvecs = sorted_eig(hami)
  eigvec = eigvecs[:,band]
  for i in eachindex(points)
    point = model.wfcs[1].points[i].p
    dip_x = zero(Complex{T})
    dip_y = zero(Complex{T})
    dip_z = zero(Complex{T})
    for i1=1:length(eigvec)
      c1 = eigvec[i1]
      w1 = k_wfcs[div(i1-1,length(k_wfcs))+1].points[i].w
      for i2=1:length(eigvec)
        w2 = k_wfcs[div(i2-1,length(k_wfcs))+1].points[i].w
        c2 = eigvec[i2]
        fac = conj(c1*w1)*c2*w2
        dip_x+=fac*point.x
        dip_y+=fac*point.y
        dip_z+=fac*point.z
      end
    end
    points[i]=(point,Point3D(real(dip_x),real(dip_y),real(dip_z)))
  end
  return points
end

function calculate_density_wfc(wfc::Wfc3D{T}) where T
  density_wfc = deepcopy(wfc)
  for (i,wfc_p) in enumerate(wfc.points)
    density_wfc.points[i] = WfcPoint3D{T}(conj(wfc_p.w)*wfc_p.w,wfc_p.p)
  end
  return density_wfc
end

function calculate_density_wfc_normalized(wfc::Wfc3D{T}) where T
  density_wfc = deepcopy(wfc)
  n=zero(Complex{T})
  for (i,wfc_p) in enumerate(wfc.points)
    density_wfc.points[i] = WfcPoint3D{T}(conj(wfc_p.w)*wfc_p.w,wfc_p.p)
    n+=norm(wfc_p.w)^2
  end
  return density_wfc/sqrt(n)
end

#-----------------------------GPU STUFF------------------------#
using CUDAdrv, CUDAnative
function bloch_kernel(wfc_orig, indices,coefficients,cell,dim, out)
  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
  i1 = (blockIdx().x-1) * blockDim().x + threadIdx().x
  i2 = (blockIdx().y-1) * blockDim().y + threadIdx().y
  i3 = (blockIdx().z-1) * blockDim().z + threadIdx().z
  
  for (c,(ind1,ind2)) in zip(coefficients,indices)
    j1 = i1-ind2[1]
    j2 = i2-ind2[2]
    j3 = i3-ind2[3]
    if i1 <= dim_a && i1>= ind1[1] &&  j1 <= dim_a && j1 > 0
      if i2 <= dim_b && i2>= ind1[2] &&  j2 <= dim_b && j2 > 0
        if i3 <= dim_c && i3>= ind1[3] &&  j3 <= dim_c && j3 > 0
          out[i1,i2,i3] += c*wfc_orig[j1,j2,j3]
        end
      end
    end
  end 
  return nothing
end

function construct_bloch_sums_gpu(wfcs::Array{CuArray{Complex{T},3},1}, k::Array{T},orig_wfc::Wfc3D{T},indices::Array{Tuple{Tuple{Int32,Int32,Int32},Tuple{Int32,Int32,Int32}},1},coefficients::Array{Complex{T},1},ctx,dev) where T <: AbstractFloat
  
  k_wfcs = Array{Wfc3D{T},1}(size(wfcs)[1])
  
  total_threads = min(length(wfcs[1]),attribute(dev,CUDAdrv.MAX_THREADS_PER_BLOCK))
  threads_x     = floor(Int, total_threads^(1/3))
  threads_y     = threads_x
  threads_z     = threads_x
  threads       = (threads_x,threads_y,threads_z)
  blocks        = Int.(ceil.(size(wfcs[1])./threads))
  
  for (n,wfc) in enumerate(wfcs)
    out = similar(wfc)
    @cuda (blocks,threads) bloch_kernel(wfc,CuArray(indices),CuArray(coefficients),CuArray(orig_wfc.cell),CuArray(Int64[size(wfc)...]),out)
    t_points = Array(out)
    t_out    = Array{WfcPoint3D{T},3}(size(wfc))
    for i in eachindex(wfc)
      t_out[i] = WfcPoint3D(t_points[i],orig_wfc.points[i].p)
    end
    k_wfcs[n] = Wfc3D(t_out,orig_wfc.cell,orig_wfc.atom)
  end
  
  return k_wfcs
end

function calculate_angmom_kernel(wfc1,wfc2,points,dim,center,V,out,n1,n2)
  dim_a = dim[1]
  dim_b = dim[2]
  dim_c = dim[3]
  center_x = center[1]
  center_y = center[2]
  center_z = center[3]
  dadx = V[1,1]
  dbdx = V[2,1]
  dcdx = V[3,1]
  dady = V[1,2]
  dbdy = V[2,2]
  dcdy = V[3,2]
  dadz = V[1,3]
  dbdz = V[2,3]
  dcdz = V[3,3]
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  i1 = (blockIdx().y-1) * blockDim().y + threadIdx().y
  i2 = (blockIdx().z-1) * blockDim().z + threadIdx().z
  
  if i > 1 && i1 > 1 && i2 > 1
    if i < dim_a && i1 < dim_b && i2 < dim_c 
      ddax = (wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2])*dadx
      ddbx = (wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2])*dbdx
      ddcx = (wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1])*dcdx
      ddx = ddax+ddbx+ddcx
      
      dday = (wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2])*dady
      ddby = (wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2])*dbdy
      ddcy = (wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1])*dcdy
      ddy = dday+ddby+ddcy
      
      ddaz = (wfc2[i+1,i1,i2]-wfc2[i-1,i1,i2])*dadz
      ddbz = (wfc2[i,i1+1,i2]-wfc2[i,i1-1,i2])*dbdz
      ddcz = (wfc2[i,i1,i2+1]-wfc2[i,i1,i2-1])*dcdz
      ddz = ddaz+ddbz+ddcz
      
      out[1] += CUDAnative.conj(wfc1[i,i1,i2])*-1im*((points[i,i1,i2][2]-center_y)*ddz-(points[i,i1,i2][3]-center_z)*ddy)
      out[2] += CUDAnative.conj(wfc1[i,i1,i2])*-1im*((points[i,i1,i2][3]-center_z)*ddx-(points[i,i1,i2][1]-center_x)*ddz)
      out[3] += CUDAnative.conj(wfc1[i,i1,i2])*-1im*((points[i,i1,i2][1]-center_x)*ddy-(points[i,i1,i2][2]-center_y)*ddx)
      sync_threads()
      # out[1] += wfc1[i,i1,i2]*-1im*((points[i,i1,i2][2]-center_y)*ddz-(points[i,i1,i2][3]-center_z)*ddy)
      # out[2] += CUDAnative.conj(wfc1[i,i1,i2].w)*-1im*((wfc2[i,i1,i2].p.z-center_z)*ddx-(wfc2[i,i1,i2].p.x-center_x)*ddz)
      # out[3] += CUDAnative.conj(wfc1[i,i1,i2].w)*-1im*((wfc2[i,i1,i2].p.x-center_x)*ddy-(wfc2[i,i1,i2].p.y-center_y)*ddx)
      # n1+= wfc1[i,i1,i2]*wfc1[i,i1,i2]
      # n2+= wfc2[i,i1,i2]*wfc2[i,i1,i2]
    end
  end
  return nothing
end
"Calculates the angular momenta between two wavefunctions, around the atom of the second wavefunction."
function calculate_angmom_gpu(wfc1::CuArray{Complex{T},3},wfc2::CuArray{Complex{T},3},points,center,V,num_threads) where T<:AbstractFloat
  total_threads = min(length(wfc1),num_threads)
  threads_x     = floor(Int, total_threads^(1/3))
  threads_y     = threads_x
  threads_z     = threads_x
  threads       = (threads_x,threads_y,threads_z)
  blocks        = Int.(ceil.(size(wfc1)./threads))
    
  n1 = CuArray([zero(Complex{T})])
  n2 = CuArray([zero(Complex{T})])
  L = CuArray([zero(Complex{T}) for i=1:3])
  dim = CuArray(Int32[size(wfc1)...])

  @cuda (blocks,threads) calculate_angmom_kernel(wfc1,wfc2,points,dim,center,CuArray(V),L,n1,n2)
  L_out = Array(L)
  # return L_out/sqrt(Array(n1)[1]*Array(n2)[1])
  return L_out
end

function calculate_angmom_gpu(wfc1::Wfc3D{T},wfc2::Wfc3D{T},num_threads) where T<:AbstractFloat
  if wfc1.atom != wfc2.atom
    return zero(Complex{T}),zero(Complex{T}),zero(Complex{T})
  else
    origin = wfc1[1,1,1].p
    a = wfc1[2,1,1].p - origin
    b = wfc1[1,2,1].p - origin
    c = wfc1[1,1,2].p - origin
    V = inv([convert(Array,a) convert(Array,b) convert(Array,c)])
    return calculate_angmom_gpu(CuArray([p.w for p in wfc1.points]),CuArray([p.w for p in wfc2.points]),CuArray([(p.p.x,p.p.y,p.p.z) for p in wfc1.points]),CuArray([wfc2.atom.center.x,wfc2.atom.center.y,wfc2.atom.center.z]),V,num_threads)
  end
end