"""
    read_xsf_file(filename::String, atom::PhysAtom, T=Float32)

Returns a Wfc3D{T} upon reading a Wannier wavefunction file. The atom specified is used in calculations such as angular momentum calculations.
"""
function read_xsf_file(filename::String, atom::PhysAtom, T=Float32)
  open(filename) do f
    primCell = zeros(T,3,3)
    while !eof(f)
      line = readline(f)
      if line == "PRIMVEC"
        line0 = split(readline(f))
        line1 = split(readline(f))
        line2 = split(readline(f))
        primCell[1,:] = map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),line0)
        primCell[2,:] = map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),line1)
        primCell[3,:] = map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),line2)
      end
      if line == " DATAGRID_3D_DENSITY"
        line0 = split(readline(f))
        line1 = split(readline(f))
        line2 = split(readline(f))
        line3 = split(readline(f))
        line4 = split(readline(f))
        nx = parse(Int,line0[1])
        ny = parse(Int,line0[2])
        nz = parse(Int,line0[3])
        origin = Point3D{T}(parse(T,line1[1]),parse(T,line1[2]),parse(T,line1[3]))
        a_vec = [parse(T,line2[1]),parse(T,line2[2]),parse(T,line2[3])]
        b_vec = [parse(T,line3[1]),parse(T,line3[2]),parse(T,line3[3])]
        c_vec = [parse(T,line4[1]),parse(T,line4[2]),parse(T,line4[3])]
        a_array = collect(T,linspace(0, 1, nx))
        b_array = collect(T,linspace(0, 1, ny))
        c_array = collect(T,linspace(0, 1, nz))
        out = Wfc3D(Array{WfcPoint3D{T},3}(nx,ny,nz),[Point3D{T}(primCell[1,:]);Point3D{T}(primCell[2,:]);Point3D{T}(primCell[3,:])],atom)
        line = readline(f)
        k=1
        k1=1
        k2=1
        while line!= "END_DATAGRID_3D"
          tmp = Array{Complex{T}}(map(x->(v = tryparse(T,x); isnull(v) ? Complex(0.0,0.0) : Complex{T}(get(v),0.0)),split(line)))
          for t in tmp
            x = origin.x+(a_vec*a_array[k])[1]+(b_vec*b_array[k1])[1]+(c_vec*c_array[k2])[1]
            y = origin.y+(a_vec*a_array[k])[2]+(b_vec*b_array[k1])[2]+(c_vec*c_array[k2])[2]
            z = origin.z+(a_vec*a_array[k])[3]+(b_vec*b_array[k1])[3]+(c_vec*c_array[k2])[3]
            out.points[k,k1,k2] = WfcPoint3D{T}(t,Point3D{T}(x,y,z))
            if k<nx
              k+=1
            else
              k=1
              k1+=1
              if k1>ny
                k1=1
                k2+=1
                if k2>nz
                  k2=1
                end
              end
            end
          end
          line = readline(f)
        end
        return out
      end
      if line == "BEGIN_DATAGRID_3D_UNKNOWN"
        line0 = split(readline(f))
        line1 = split(readline(f))
        line2 = split(readline(f))
        line3 = split(readline(f))
        line4 = split(readline(f))
        nx = parse(Int,line0[1])
        ny = parse(Int,line0[2])
        nz = parse(Int,line0[3])
        origin = Point3D{T}(parse(T,line1[1]),parse(T,line1[2]),parse(T,line1[3]))
        a_vec = [parse(T,line2[1]),parse(T,line2[2]),parse(T,line2[3])]
        b_vec = [parse(T,line3[1]),parse(T,line3[2]),parse(T,line3[3])]
        c_vec = [parse(T,line4[1]),parse(T,line4[2]),parse(T,line4[3])]
        a_array = collect(T,linspace(0, 1, nx))
        b_array = collect(T,linspace(0, 1, ny))
        c_array = collect(T,linspace(0, 1, nz))
        out = Wfc3D(Array{WfcPoint3D{T},3}(nx,ny,nz),[Point3D(primCell[1,:]);Point3D(primCell[2,:]);Point3D(primCell[3,:])],atom)
        line = readline(f)
        k=1
        k1=1
        k2=1
        while line!= "END_DATAGRID_3D"
          tmp = Array{Complex{T}}(map(x->(v = tryparse(T,x); isnull(v) ? Complex(0.0,0.0) : Complex{T}(get(v),0.0)),split(line)))
          for t in tmp
            
            x = origin.x+(a_vec*a_array[k])[1]+(b_vec*b_array[k1])[1]+(c_vec*c_array[k2])[1]
            y = origin.y+(a_vec*a_array[k])[2]+(b_vec*b_array[k1])[2]+(c_vec*c_array[k2])[2]
            z = origin.z+(a_vec*a_array[k])[3]+(b_vec*b_array[k1])[3]+(c_vec*c_array[k2])[3]
            out.points[k,k1,k2] = WfcPoint3D{T}(t,Point3D{T}(x,y,z))
            if k<nx
              k+=1
            else
              k=1
              k1+=1
              if k1>ny
                k1=1
                k2+=1
                if k2>nz
                  k2=1
                end
              end
            end
          end
          line = readline(f)
        end
        return out
      end
    end
  end
end

"Returns a Wfc3D{T} that is centered around an atom at the origin with 0 soc strength."
read_xsf_file(filename::String, T=Float32) = read_xsf_file(filename,PhysAtom(),T)

"""
    write_xsf_file(filename::String, wfc::Wfc3D{T}) where T<:AbstractFloat


Writes the real part of the Wfc3D to a .xsf file that is readable by XCrysden or VESTA.
"""
function write_xsf_file(filename::String,wfc::Wfc3D{T}) where T<:AbstractFloat
  open(filename,"w") do f
    origin = wfc.points[1,1,1].p
    write(f,["# Generated from PhD calculations\n", "", "BEGIN_BLOCK_DATAGRID_3D\n", "3D_FIELD\n",
                      "BEGIN_DATAGRID_3D_UNKNOWN\n"])
    write(f,"$(size(wfc.points)[1])    $(size(wfc.points)[2])     $(size(wfc.points)[3])\n")
    write(f,"$(origin.x)   $(origin.y)   $(origin.z)\n")
    write(f,"$(wfc.points[end,1,1].p.x-origin.x)   $(wfc.points[end,1,1].p.y-origin.y)   $(wfc.points[end,1,1].p.z-origin.z)\n")
    write(f,"$(wfc.points[1,end,1].p.x-origin.x)   $(wfc.points[1,end,1].p.y-origin.y)   $(wfc.points[1,end,1].p.z-origin.z)\n")
    write(f,"$(wfc.points[1,1,end].p.x-origin.x)   $(wfc.points[1,1,end].p.y-origin.y)   $(wfc.points[1,1,end].p.z-origin.z)\n")
    for wfp in wfc.points
      write(f,"$(real(wfp.w)) ")
    end
    write(f,"\n")
    write(f,["END_DATAGRID_3D\n", "END_BLOCK_DATAGRID_3D\n"])
  end
end

"""
    read_hami_file(filename::String, T=Float32)

Returns an array of tuples that define the hopping parameters of the Wannier Tight Binding Hamiltonian.
"""
function read_hami_file(filename::String, T=Float32)
  open(filename) do f
    out = Array{Tuple{Int,Int,Int,Int,Int,Complex{T}},1}()
    degen = Int64[]
    line_nr = 0
    readline(f)
    n_wanfun = parse(Int64,readline(f))
    readline(f)
    while !eof(f)
      l = split(readline(f))
      if length(l)==7
        line_nr+=1
        ints = [parse(Int,x) for x in l[1:5]]
        complex = Complex{T}(parse(T,l[6]),parse(T,l[7]))/degen[div(line_nr-1,n_wanfun^2)+1]
        push!(out,(ints...,complex))
      elseif length(l)!=7
        ints = parse.(Int,l)
        push!(degen,ints...)
      end
    end
    return out
  end
end

"""
    read_dipole_file(filename::String, T=Float32)

Returns and array of tuples that define the dipoles between the Wannier functions in different unit cells.
"""
function read_dipole_file(filename::String, T=Float32)
  open(filename) do  f
    out = Array{Tuple{Int,Int,Int,Int,Int,Point3D{T}},1}()
    readline(f)
    readline(f)
    while !eof(f)
      l= split(readline(f))
      ints = [parse(Int,x) for x in l[1:5]]
      dipole = Point3D(parse(T,l[6]),parse(T,l[8]),parse(T,l[10]))
      push!(out,(ints...,dipole))
    end
    return out
  end
end



#stuff that is not used currently
"""
    read_potential_file(filename::String, T=Float32)

Reads a Quantum Espresso potential output file. 
"""
function read_potential_file(filename::String, T=Float32)
  tmp_data=nothing
  open(filename) do f
    primCell = zeros(T,3,3)
    while !eof(f)
      line = readline(f)
      if line == "PRIMVEC"
        line0 = split(readline(f))
        line1 = split(readline(f))
        line2 = split(readline(f))
        primCell[1,:] = map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),line0)
        primCell[2,:] = map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),line1)
        primCell[3,:] = map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),line2)
      end
      if line == "DATAGRID_3D_UNKNOWN"
        line0 = split(readline(f))
        line1 = split(readline(f))
        line2 = split(readline(f))
        line3 = split(readline(f))
        line4 = split(readline(f))
        nx = parse(Int,line0[1])
        ny = parse(Int,line0[2])
        nz = parse(Int,line0[3])
        tmp_data=Array{T,3}(nx,ny,nz)
        line = readline(f)
        k=1
        k1=1
        k2=1
        while line!= "END_DATAGRID_3D"
          tmp = Array{T}(map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),split(line)))
          for t in tmp
            tmp_data[k,k1,k2] = t
            if k<nx
              k+=1
            else
              k=1
              k1+=1
              if k1>ny
                k1=1
                k2+=1
                if k2>nz
                  k2=1
                end
              end
            end
          end
          line = readline(f)
        end
        return interpolate(tmp_data,BSpline(Cubic(Line())),OnGrid())
      end
    end
  end
end