@recipe function f(band::WannierBand,data=:eigvals;ks=nothing,fermi=0,linewidth=2)
  if ks==nothing
    ks=[]
    k_m = band.k_points[div(size(band.k_points)[1]+1,2)]
    for k in band.k_points
      push!(ks,norm(k-k_m))
    end
  end
  ks[1:div(length(ks),2)]=-ks[1:div(length(ks),2)]
  # if ks==nothing
  #   ks = 1:size(band.k_points)[1]
  # end
  if fermi != 0
    band = apply_fermi_level(band,fermi)
  end
  linewidth --> linewidth
  out = []
  if data==:eigvals
    out = band.eigvals
    title --> "Eigenvalues"
    yguide -->(haskey(d,:yguide) ? d[:yguide] : "energy (eV)")
  else
    if data==:cm_x
      title --> "Center of Mass (X)"
      yguide --> "Rx (Angström)"
      for cm in band.cms
        push!(out,cm.x)
      end
    elseif data==:cm_y
      title --> "Center of Mass (Y)"
      yguide --> "Ry (Angström)"
      for cm in band.cms
        push!(out,cm.y)
      end
    elseif data==:cm_z
      title --> "Center of Mass (Z)"
      yguide --> L"R_z"*" ("*L"\AA"*")"
      for cm in band.cms
        push!(out,cm.z)
      end
    elseif data==:angmom_x
      title --> "Orbital Angular Momentum (X)"
      yguide --> L"L_x (arb. units)"
      for angmom in band.angmoms
        push!(out,angmom[1].x+angmom[2].x)
      end
    elseif data==:angmom_y
      title --> "Orbital Angular Momentum (Y)"
      yguide --> L"L_y (arb. units)"
      for angmom in band.angmoms
        push!(out,angmom[1].y+angmom[2].y)
      end
    elseif data==:angmom_z
      title --> "Orbital Angular Momentum (Z)"
      yguide --> "Lz (arb. units)"
      for angmom in band.angmoms
        push!(out,angmom[1].z+angmom[2].z)
      end
    elseif data==:angmom1_x
      title --> "Orbital Angular Momentum (X)"
      yguide --> "Lx (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[1].x)
      end
    elseif data==:angmom1_y
      title --> "Orbital Angular Momentum (Y)"
      yguide --> "Ly (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[1].y)
      end
    elseif data==:angmom1_z
      title --> "Orbital Angular Momentum (Z)"
      yguide --> "Lz (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[1].z)
      end
    elseif data==:angmom2_x
      title --> "Orbital Angular Momentum (X)"
      yguide --> L"L_x"*" (arb. units)"
      label --> "OAM around Te"
      for angmom in band.angmoms
        push!(out,angmom[2].x)
      end
    elseif data==:angmom2_y
      title --> "Orbital Angular Momentum (Y)"
      yguide --> L"L_y"*" (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[2].y)
      end
    elseif data==:angmom2_z
      title --> "Orbital Angular Momentum (Z)"
      yguide --> "Lz (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[2].z)
      end
    elseif data==:angmom2_xy
      title --> "Orbital Angular Momentum (XY)"
      yguide --> "norm(Lx)+norm(Ly)"
      label --> "Total angmom"
      for (spin,angmom) in zip(band.spins,band.angmoms)
        push!(out,sqrt((angmom[2].x+spin[2].x)^2+(angmom[2].y+spin[2].y)^2))
      end
    elseif data==:spin1_x
      title --> "Spin Angular Momentum (X)"
      yguide --> "Sx (arb. units)"
      label --> "SAM around Ge"
      for spin in band.spins
        push!(out,spin[1].x)
      end
    elseif data==:spin1_y
      title --> "Spin Angular Momentum (Y)"
      yguide --> "Sy (arb. units)"
      label --> "SAM around Ge"
      for spin in band.spins
        push!(out,spin[1].y)
      end
    elseif data==:spin1_z
      title --> "Spin Angular Momentum (Z)"
      yguide --> "Sz (arb. units)"
      label --> "SAM around Ge"
      for spin in band.spins
        push!(out,spin[1].z)
      end
    elseif data==:spin2_x
      title --> "Spin Angular Momentum (X)"
      yguide --> L"S_x"*" (arb. units)"
      label --> "SAM around Te"
      for spin in band.spins
        push!(out,spin[2].x)
      end
    elseif data==:spin2_y
      title --> "Spin Angular Momentum (Y)"
      yguide --> L"S_y"*" (arb. units)"
      label --> "SAM around Te"
      for spin in band.spins
        push!(out,spin[2].y)
      end
    elseif data==:spin2_z
      title --> "Spin Angular Momentum (Z)"
      yguide --> "Sz (arb. units)"
      label --> "SAM around Te"
      for spin in band.spins
        push!(out,spin[2].z)
      end
    elseif data==:epot
      title --> "Electrostatic Potential"
      yguide --> "E (arb. units)"
      for epot in band.epots
        push!(out,epot)
      end
    end
  end
  ks,out
end

@recipe function f(::Type{WannierBand},band::WannierBand,data=:eigvals;ks=nothing,fermi=0,linewidth=2)
  if ks==nothing
    ks=[]
    k_m = band.k_points[div(size(band.k_points)[1]+1,2)]
    for k in band.k_points
      push!(ks,norm(k-k_m))
    end
  end
  ks[1:div(length(ks),2)]=-ks[1:div(length(ks),2)]
  # if ks==nothing
  #   ks = 1:size(band.k_points)[1]
  # end
  if fermi != 0
    band = apply_fermi_level(band,fermi)
  end
  linewidth --> linewidth
  out = []
  if data==:eigvals
    out = band.eigvals
    title --> "Eigenvalues"
    yguide -->(haskey(d,:yguide) ? d[:yguide] : "energy (eV)")
  else
    if data==:cm_x
      title --> "Center of Mass (X)"
      yguide --> "Rx (Angström)"
      for cm in band.cms
        push!(out,cm.x)
      end
    elseif data==:cm_y
      title --> "Center of Mass (Y)"
      yguide --> "Ry (Angström)"
      for cm in band.cms
        push!(out,cm.y)
      end
    elseif data==:cm_z
      title --> "Center of Mass (Z)"
      yguide --> L"R_z"*" ("*L"\AA"*")"
      for cm in band.cms
        push!(out,cm.z)
      end
    elseif data==:angmom_x
      title --> "Orbital Angular Momentum (X)"
      yguide --> L"L_x (arb. units)"
      for angmom in band.angmoms
        push!(out,angmom[1].x+angmom[2].x)
      end
    elseif data==:angmom_y
      title --> "Orbital Angular Momentum (Y)"
      yguide --> L"L_y (arb. units)"
      for angmom in band.angmoms
        push!(out,angmom[1].y+angmom[2].y)
      end
    elseif data==:angmom_z
      title --> "Orbital Angular Momentum (Z)"
      yguide --> "Lz (arb. units)"
      for angmom in band.angmoms
        push!(out,angmom[1].z+angmom[2].z)
      end
    elseif data==:angmom1_x
      title --> "Orbital Angular Momentum (X)"
      yguide --> "Lx (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[1].x)
      end
    elseif data==:angmom1_y
      title --> "Orbital Angular Momentum (Y)"
      yguide --> "Ly (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[1].y)
      end
    elseif data==:angmom1_z
      title --> "Orbital Angular Momentum (Z)"
      yguide --> "Lz (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[1].z)
      end
    elseif data==:angmom2_x
      title --> "Orbital Angular Momentum (X)"
      yguide --> L"L_x"*" (arb. units)"
      label --> "OAM around Te"
      for angmom in band.angmoms
        push!(out,angmom[2].x)
      end
    elseif data==:angmom2_y
      title --> "Orbital Angular Momentum (Y)"
      yguide --> L"L_y"*" (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[2].y)
      end
    elseif data==:angmom2_z
      title --> "Orbital Angular Momentum (Z)"
      yguide --> "Lz (arb. units)"
      label --> "OAM around Ge"
      for angmom in band.angmoms
        push!(out,angmom[2].z)
      end
    elseif data==:angmom2_xy
      title --> "Orbital Angular Momentum (XY)"
      yguide --> "norm(Lx)+norm(Ly)"
      label --> "Total angmom"
      for (spin,angmom) in zip(band.spins,band.angmoms)
        push!(out,sqrt((angmom[2].x+spin[2].x)^2+(angmom[2].y+spin[2].y)^2))
      end
    elseif data==:spin1_x
      title --> "Spin Angular Momentum (X)"
      yguide --> "Sx (arb. units)"
      label --> "SAM around Ge"
      for spin in band.spins
        push!(out,spin[1].x)
      end
    elseif data==:spin1_y
      title --> "Spin Angular Momentum (Y)"
      yguide --> "Sy (arb. units)"
      label --> "SAM around Ge"
      for spin in band.spins
        push!(out,spin[1].y)
      end
    elseif data==:spin1_z
      title --> "Spin Angular Momentum (Z)"
      yguide --> "Sz (arb. units)"
      label --> "SAM around Ge"
      for spin in band.spins
        push!(out,spin[1].z)
      end
    elseif data==:spin2_x
      title --> "Spin Angular Momentum (X)"
      yguide --> L"S_x"*" (arb. units)"
      label --> "SAM around Te"
      for spin in band.spins
        push!(out,spin[2].x)
      end
    elseif data==:spin2_y
      title --> "Spin Angular Momentum (Y)"
      yguide --> L"S_y"*" (arb. units)"
      label --> "SAM around Te"
      for spin in band.spins
        push!(out,spin[2].y)
      end
    elseif data==:spin2_z
      title --> "Spin Angular Momentum (Z)"
      yguide --> "Sz (arb. units)"
      label --> "SAM around Te"
      for spin in band.spins
        push!(out,spin[2].z)
      end
    elseif data==:epot
      title --> "Electrostatic Potential"
      yguide --> "E (arb. units)"
      for epot in band.epots
        push!(out,epot)
      end
    end
  end
  ks,out
end

@recipe function f(bands::Array{<:WannierBand,1},data::Array{Symbol,1})
  layout := @layout(length(data))
  for (i,dat) in enumerate(data)
    @series begin
      subplot := i
      bands,data[i]
    end
  end
end

@recipe function f(dfbands::Array{<:Band,1},tbbands::Array{<:Band,1})
  @series begin
    label--> "DFT Calculation"
    line--> (1,1.0,:blue)
    dfbands[1],:eigvals
  end
  @series begin
    label--> "Wannier Interpolation"
    line--> (2,:dot,1.0,:red)
    tbbands[1],:eigvals
  end
  for band in dfbands[2:end]
    @series begin
      label--> ""
      line--> (1,1.0,:blue)
      band,:eigvals
    end
  end
  for band in tbbands[2:end]
    @series begin
      label--> ""
      line--> (2,:dot,1.0,:red)
      band,:eigvals
    end
  end
end