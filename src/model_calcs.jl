function_names = ["calculate_eig_soc","calculate_eig_pot","calculate_eig_angmom","calculate_eig_pot_angmom","calculate_eig_pot_soc","calculate_eig_angmom_soc","calculate_eig_pot_angmom_soc",
               "calculate_eig_cm","calculate_eig_cm_soc","calculate_eig_cm_pot","calculate_eig_cm_pot_soc","calculate_eig_cm_angmom","calculate_eig_cm_angmom_soc","calculate_eig_cm_pot_angmom","calculate_eig_cm_pot_angmom_soc"]

begin
  cm_tcalc = :(tmp_cm = calculate_k_dips(model.dip_raw,k))
  pot_tcalc = :(tmp_pot = calculate_tmp_pot(model.wfcs,pot))
  angmom_precalc = quote tmp_angmom = calculate_angmoms(model.wfcs)::Array{Array{Complex{T},1},2}
                       tmp_spin_x::Array{Complex{T},2},tmp_spin_y::Array{Complex{T},2},tmp_spin_z::Array{Complex{T},2} = calculate_spins(model.wfcs)
                         end
  cm_cinit = quote eigcm_x = zero(Complex{T})
                   eigcm_y = zero(Complex{T})
                   eigcm_z = zero(Complex{T}) end
  angmom_cinit = quote eigmom = [[zero(Complex{T}) for n=1:3] for m=1:length(atoms)]
                       eigspin = [[zero(Complex{T}) for n=1:3] for m=1:length(atoms)] end
  pot_cinit = :(eigpot = zero(Complex{T}))
  cm_calc = quote eigcm_x += conj(c1)*c2*tmp_cm[i1,i2].x
                  eigcm_y += conj(c1)*c2*tmp_cm[i1,i2].y
                  eigcm_z += conj(c1)*c2*tmp_cm[i1,i2].z end
  pot_calc = :(eigpot += conj(c1)*c2*tmp_pot[i1,i2])
  angmom_calc = quote eigmom[atom_indices[i1]] += conj(c1)*c2*tmp_angmom[i1,i2]
                      eigspin[atom_indices[i1]][1] += conj(c1)*c2*tmp_spin_x[i1,i2]
                      eigspin[atom_indices[i1]][2] += conj(c1)*c2*tmp_spin_y[i1,i2]
                      eigspin[atom_indices[i1]][3] += conj(c1)*c2*tmp_spin_z[i1,i2]
                      end
  cm_loop_end = :(out_bands[i].cms[j] = Point3D{T}(norm(eigcm_x),norm(eigcm_y),norm(eigcm_z)))
  pot_loop_end = :(out_bands[i].epots[j] = real(eigpot))
  angmom_loop_end = quote 
                          for n=1:length(atoms)
                            out_bands[i].angmoms[j][n] = Point3D{T}(real(eigmom[n][1]),real(eigmom[n][2]),real(eigmom[n][3]))
                            out_bands[i].spins[j][n] = Point3D{T}(real(eigspin[n][1]),real(eigspin[n][2]),real(eigspin[n][3]))
                          end
                        end
  for name in function_names 
    func_vars = []
    tmp_calcs = quote end
    calc_init = quote end
    calcs     = quote end
    loop_end  = quote end
    if contains(name,"soc")
      intro = quote out_bands = [WannierBand(Array{T,1}(size(k_points)[1]),Array{Array{Complex{T},1},1}(length(k_points)),Array{Point3D{T},1}(size(k_points)[1]),[[Point3D{T}(0.0) for j=1:length(atoms)] for n=1:length(k_points)],[[Point3D{T}(0.0) for j=1:length(atoms)] for n=1:length(k_points)],k_points) for i=1:size(model.wfcs)[1]*2] end
      push!(intro.args,angmom_precalc)
      ham_line = :(hami = construct_soc_hami(hami_from_k(model.hami_raw,k),tmp_angmom,map(x->x.atom,model.wfcs)))
    else
      intro = quote out_bands = [WannierBand(Array{T,1}(size(k_points)[1]),Array{Array{Complex{T},1},1}(length(k_points)),Array{Point3D{T},1}(size(k_points)[1]),[[Point3D{T}(0.0) for j=1:length(atoms)] for n=1:length(k_points)],[[Point3D{T}(0.0) for j=1:length(atoms)] for n=1:length(k_points)],k_points) for i=1:size(model.wfcs)[1]] end
      ham_line = :(hami = hami_from_k(model.hami_raw,k))
    end
    if contains(name,"pot")
      push!(intro.args,:(pot = [T(potential[i1,i2,i3]) for i1=1:45/26:46,i2=1:45/26:46,i3=1:45/26:46]))
      func_vars = [:potential]
      push!(tmp_calcs.args,pot_tcalc)
      push!(calc_init.args,pot_cinit)
      push!(calcs.args,pot_calc)
      push!(loop_end.args,pot_loop_end)
    end
    if contains(name,"angmom")
      if !contains(name,"soc")
        push!(intro.args,angmom_precalc)
      end
      push!(calc_init.args,angmom_cinit)
      push!(calcs.args,angmom_calc)
      push!(loop_end.args,angmom_loop_end)
    end
    if contains(name,"cm")
      push!(tmp_calcs.args,cm_tcalc)
      push!(calc_init.args,cm_cinit)
      push!(calcs.args,cm_calc)
      push!(loop_end.args,cm_loop_end)
    end
    @eval function $(Symbol(name)){T<:AbstractFloat}(model::WannierModel{T},$(func_vars...),k_points::Array{Array{T,1},1})
            atoms = PhysAtom[]
            atom_indices = Int[]
            for wfc in model.wfcs
              if !in(wfc.atom,atoms)
                push!(atoms,wfc.atom)
                push!(atom_indices,length(atoms))
              else
                push!(atom_indices,findfirst(atoms,wfc.atom))
              end
            end
            atom_indices = [atom_indices;atom_indices]
            $intro
            Threads.@threads for j=1:size(k_points)[1]
              k = k_points[j]
              $ham_line
              eigvals,eigvectors = sorted_eig(hami)
              $tmp_calcs
              for i=1:size(eigvals)[1]
                eigval = eigvals[i]
                eigvec = eigvectors[:,i]
                $calc_init
                for i1=1:size(eigvec)[1]
                  c1 = eigvec[i1]
                  for i2=1:size(eigvec)[1]
                    c2 = eigvec[i2]
                    $calcs
                  end
                end
                out_bands[i].eigvals[j] = real(eigval)
                out_bands[i].eigvec[j]=eigvec
                $loop_end
              end
            end
            return out_bands
          end
    @eval $(parse(name))(model,$(func_vars...)) = $(parse(name))(model,$(func_vars...),model.k_points)
    @eval function$(parse(name))(model,$(func_vars...),k_range::StepRangeLen)
      mid = div(size(model.k_points)[1],2)+1
      beg = Int64(k_range[1])
      steps = div(size(k_range)[1],2)
      last = Int64(k_range[end])
      kxs = [linspace(model.k_points[beg][1],model.k_points[mid][1],steps)...,model.k_points[mid][1] ,linspace(model.k_points[mid][1],model.k_points[last][1],steps)[2:end]...]
      kys = [linspace(model.k_points[beg][2],model.k_points[mid][2],steps)...,model.k_points[mid][2] ,linspace(model.k_points[mid][2],model.k_points[last][2],steps)[2:end]...]
      kzs = [linspace(model.k_points[beg][3],model.k_points[mid][3],steps)...,model.k_points[mid][3] ,linspace(model.k_points[mid][3],model.k_points[last][3],steps)[2:end]...]
      kxs_t = [linspace(model.k_points[beg][1],model.k_points[mid][1],steps*100)... ,linspace(model.k_points[mid][1],model.k_points[last][1],steps*100)[2:end]...]
      kys_t = [linspace(model.k_points[beg][2],model.k_points[mid][2],steps*100)... ,linspace(model.k_points[mid][2],model.k_points[last][2],steps*100)[2:end]...]
      kzs_t = [linspace(model.k_points[beg][3],model.k_points[mid][3],steps*100)... ,linspace(model.k_points[mid][3],model.k_points[last][3],steps*100)[2:end]...]
      # kxs[div(length(kxs),2)]+=0.00001
      kxs[div(length(kxs),2)] = kxs_t[div(length(kxs_t),2)]
      kxs[div(length(kxs),2)+1] = kxs_t[div(length(kxs_t),2)+2]
      kys[div(length(kxs),2)] = kys_t[div(length(kxs_t),2)]
      kys[div(length(kxs),2)+1] = kys_t[div(length(kxs_t),2)+2]
      kzs[div(length(kxs),2)] = kzs_t[div(length(kxs_t),2)]
      kzs[div(length(kxs),2)+1] = kzs_t[div(length(kxs_t),2)+2]
      k_points = [[kxs[i],kys[i],kzs[i]] for i=1:length(kxs)]
      return $(parse(name))(model,$(func_vars...),k_points)
    end
    func = Symbol(name)
    eval(:(export $func))
  end
end

#Same as above but with bloch sums
begin
  angmom_tcalc = quote tmp_angmom = calculate_angmoms(k_wfcs)
                          tmp_spin_x,tmp_spin_y,tmp_spin_z = calculate_spins(model.wfcs)
                         end
  cm_tcalc = quote tmp_cm = calculate_k_dips(model.dip_raw,k) end
  pot_tcalc = :(tmp_pot = calculate_tmp_pot(k_wfcs,pot))
  cm_cinit = quote eigcm_x = zero(Complex{T})
    eigcm_y = zero(Complex{T})
    eigcm_z = zero(Complex{T}) end
angmom_cinit = quote eigmom = [[zero(Complex{T}) for n=1:3] for m=1:length(atoms)]
        eigspin = [[zero(Complex{T}) for n=1:3] for m=1:length(atoms)] end
pot_cinit = :(eigpot = zero(Complex{T}))
cm_calc = quote eigcm_x += conj(c1)*c2*tmp_cm[i1,i2].x
   eigcm_y += conj(c1)*c2*tmp_cm[i1,i2].y
   eigcm_z += conj(c1)*c2*tmp_cm[i1,i2].z end
pot_calc = :(eigpot += conj(c1)*c2*tmp_pot[i1,i2])
angmom_calc = quote eigmom[atom_indices[i1]] += conj(c1)*c2*tmp_angmom[i1,i2]
       eigspin[atom_indices[i1]][1] += conj(c1)*c2*tmp_spin_x[i1,i2]
       eigspin[atom_indices[i1]][2] += conj(c1)*c2*tmp_spin_y[i1,i2]
       eigspin[atom_indices[i1]][3] += conj(c1)*c2*tmp_spin_z[i1,i2]
       end
cm_loop_end = :(out_bands[i].cms[j] = Point3D{T}(norm(eigcm_x),norm(eigcm_y),norm(eigcm_z)))
pot_loop_end = :(out_bands[i].epots[j] = real(eigpot))
angmom_loop_end = quote 
           for n=1:length(atoms)
             out_bands[i].angmoms[j][n] = Point3D{T}(real(eigmom[n][1]),real(eigmom[n][2]),real(eigmom[n][3]))
             out_bands[i].spins[j][n] = Point3D{T}(real(eigspin[n][1]),real(eigspin[n][2]),real(eigspin[n][3]))
           end
         end
  for name in function_names

    func_vars = []
    tmp_calcs = quote end
    calc_init = quote end
    calcs = quote end
    loop_end = quote end
    if contains(name,"soc")
      intro = quote out_bands = [WannierBand(Array{T,1}(size(k_points)[1]),Array{Array{Complex{T},1},1}(length(k_points)),Array{Point3D{T},1}(size(k_points)[1]),[[Point3D{T}(0.0),Point3D{T}(0.0)] for i=1:length(k_points)],[[Point3D{T}(0.0),Point3D{T}(0.0)] for i=1:length(k_points)],k_points) for i=1:size(model.wfcs)[1]*2] end
      ham_line = :(hami = construct_soc_hami(hami_from_k(model.hami_raw,k),tmp_angmom,map(x->x.atom,model.wfcs)))
    else
      intro = quote out_bands = [WannierBand(Array{T,1}(size(k_points)[1]),Array{Array{Complex{T},1},1}(length(k_points)),Array{Point3D{T},1}(size(k_points)[1]),[[Point3D{T}(0.0),Point3D{T}(0.0)] for i=1:length(k_points)],[[Point3D{T}(0.0),Point3D{T}(0.0)] for i=1:length(k_points)],k_points) for i=1:size(model.wfcs)[1]] end
      ham_line = :(hami = hami_from_k(model.hami_raw,k))
    end
    if contains(name,"pot")
      push!(intro.args,:(pot = [T(potential[i1,i2,i3]) for i1=1:45/26:46,i2=1:45/26:46,i3=1:45/26:46]))
      func_vars = [:potential]
      push!(tmp_calcs.args,pot_tcalc)
      push!(calc_init.args,pot_cinit)
      push!(calcs.args,pot_calc)
      push!(loop_end.args,pot_loop_end)
    end
    if contains(name,"angmom")
      push!(tmp_calcs.args,angmom_tcalc)
      push!(calc_init.args,angmom_cinit)
      push!(calcs.args,angmom_calc)
      push!(loop_end.args,angmom_loop_end)
    end
    if contains(name,"cm")
      push!(tmp_calcs.args,cm_tcalc)
      push!(calc_init.args,cm_cinit)
      push!(calcs.args,cm_calc)
      push!(loop_end.args,cm_loop_end)
    end
    @eval function $(Symbol(name*"_bloch")){T<:AbstractFloat}(model::WannierModel{T},$(func_vars...),k_points::Array{Array{T,1},1})
            atoms = PhysAtom[]
            atom_indices = Int[]
            for wfc in model.wfcs
              if !in(wfc.atom,atoms)
                push!(atoms,wfc.atom)
                push!(atom_indices,length(atoms))
              else
                push!(atom_indices,findfirst(atoms,wfc.atom))
              end
            end
            atom_indices = [atom_indices;atom_indices]
            $intro
            Threads.@threads for j=1:size(k_points)[1]
              
              k = k_points[j]
              k_wfcs = Array{Wfc3D{T},1}(size(model.wfcs)[1])
              for (i,wfc) in enumerate(model.wfcs)
                k_wfcs[i] = construct_bloch_sum(wfc,k)
              end
              $tmp_calcs
              $ham_line
              eigvals,eigvectors = sorted_eig(hami)
              for i=1:size(eigvals)[1]
                eigval = eigvals[i]
                eigvec = eigvectors[:,i]
                $calc_init
                for i1=1:size(eigvec)[1]
                  c1 = eigvec[i1]
                  for i2=1:size(eigvec)[1]
                    c2 = eigvec[i2]
                    $calcs
                  end
                end
                out_bands[i].eigvals[j] = real(eigval)
                out_bands[i].eigvec[j]=eigvec
                $loop_end
              end
            end
            return out_bands
          end
    @eval $(parse(name*"_bloch"))(model,$(func_vars...)) = $(parse(name*"_bloch"))(model,$(func_vars...),model.k_points)
    @eval function$(parse(name*"_bloch"))(model,$(func_vars...),k_range::StepRangeLen)
      mid = div(size(model.k_points)[1],2)+1
      beg = Int64(k_range[1])
      steps = div(size(k_range)[1],2)
      last = Int64(k_range[end])
      kxs = [linspace(model.k_points[beg][1],model.k_points[mid][1],steps)...,model.k_points[mid][1] ,linspace(model.k_points[mid][1],model.k_points[last][1],steps)[2:end]...]
      kys = [linspace(model.k_points[beg][2],model.k_points[mid][2],steps)...,model.k_points[mid][2] ,linspace(model.k_points[mid][2],model.k_points[last][2],steps)[2:end]...]
      kzs = [linspace(model.k_points[beg][3],model.k_points[mid][3],steps)...,model.k_points[mid][3] ,linspace(model.k_points[mid][3],model.k_points[last][3],steps)[2:end]...]
      kxs_t = [linspace(model.k_points[beg][1],model.k_points[mid][1],steps*100)... ,linspace(model.k_points[mid][1],model.k_points[last][1],steps*100)[2:end]...]
      kys_t = [linspace(model.k_points[beg][2],model.k_points[mid][2],steps*100)... ,linspace(model.k_points[mid][2],model.k_points[last][2],steps*100)[2:end]...]
      kzs_t = [linspace(model.k_points[beg][3],model.k_points[mid][3],steps*100)... ,linspace(model.k_points[mid][3],model.k_points[last][3],steps*100)[2:end]...]
      # kxs[div(length(kxs),2)]+=0.00001
      kxs[div(length(kxs),2)] = kxs_t[div(length(kxs_t),2)]
      kxs[div(length(kxs),2)+1] = kxs_t[div(length(kxs_t),2)+2]
      kys[div(length(kxs),2)] = kys_t[div(length(kxs_t),2)]
      kys[div(length(kxs),2)+1] = kys_t[div(length(kxs_t),2)+2]
      kzs[div(length(kxs),2)] = kzs_t[div(length(kxs_t),2)]
      kzs[div(length(kxs),2)+1] = kzs_t[div(length(kxs_t),2)+2]
      k_points = [[kxs[i],kys[i],kzs[i]] for i=1:length(kxs)]
      return $(parse(name*"_bloch"))(model,$(func_vars...),k_points)
    end
    func = Symbol(name*"_bloch")
    eval(:(export $func))
  end
end
