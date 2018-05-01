#does this really need the checking for corruption stuff?
"""
read_xsf_file(filename::String, atom::Atom, T=Float64)

Returns a Wfc3D{T} upon reading a Wannier wavefunction file. The atom specified is used in calculations such as angular momentum calculations.
"""
function read_xsf_file(filename::String, T=Float64)
    open(filename) do f
        while !eof(f)
            line = readline(f)
            if line == "PRIMVEC"
                cell  = [Point3{T}(map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),split(readline(f)))) for i=1:3]
            end

            if line == " DATAGRID_3D_DENSITY" || contains(line, "DATAGRID_3D_UNKNOWN")
                nx, ny, nz = parse.(Int,split(readline(f)))
                origin     = Point3{T}(parse.(T,split(readline(f))))
                a_vec      = parse.(T,split(readline(f)))
                b_vec      = parse.(T,split(readline(f)))
                c_vec      = parse.(T,split(readline(f)))
                a_array    = collect(T,linspace(0, 1, nx))
                b_array    = collect(T,linspace(0, 1, ny))
                c_array    = collect(T,linspace(0, 1, nz))
                out        = Array{WfcPoint3{T},3}(nx,ny,nz)
                line       = readline(f)

                k  = 1
                k1 = 1
                k2 = 1
                while line != "END_DATAGRID_3D"
                    #uncomment this line if there is data corruption
                    # tmp = Array{Complex{T}}(map(x->(v = tryparse(T,x); isnull(v) ? Complex(0.0,0.0) : Complex{T}(get(v),0.0)),split(line)))
                    for t in map(x->Complex{T}(x,zero(T)),parse.(T,split(line)))
                        x = origin[1] + (a_vec * a_array[k])[1] + (b_vec * b_array[k1])[1] + (c_vec * c_array[k2])[1]
                        y = origin[2] + (a_vec * a_array[k])[2] + (b_vec * b_array[k1])[2] + (c_vec * c_array[k2])[2]
                        z = origin[3] + (a_vec * a_array[k])[3] + (b_vec * b_array[k1])[3] + (c_vec * c_array[k2])[3]
                        out[k,k1,k2] = WfcPoint3{T}(t,Point3{T}(x,y,z))
                        if k < nx
                            k += 1
                        else
                            k  = 1
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
        write(f,"$(origin[1])   $(origin[2])   $(origin[3])\n")
        write(f,"$(wfc.points[end,1,1].p[1]-origin[1])   $(wfc.points[end,1,1].p[2]-origin[2])   $(wfc.points[end,1,1].p[3]-origin[3])\n")
        write(f,"$(wfc.points[1,end,1].p[1]-origin[1])   $(wfc.points[1,end,1].p[2]-origin[2])   $(wfc.points[1,end,1].p[3]-origin[3])\n")
        write(f,"$(wfc.points[1,1,end].p[1]-origin[1])   $(wfc.points[1,1,end].p[2]-origin[2])   $(wfc.points[1,1,end].p[3]-origin[3])\n")
        for wfp in wfc.points
            write(f,"$(real(wfp.w)) ")
        end
        write(f,"\n")
        write(f,["END_DATAGRID_3D\n", "END_BLOCK_DATAGRID_3D\n"])
    end
end

"""
read_hami_file(filename::String,structure::AbstractStructure{T})

Returns an array of tuples that define the hopping parameters of the Wannier Tight Binding Hamiltonian.
"""
function read_hami_file(filename::String, structure::AbstractStructure{T}) where  T
    open(filename) do f
        out = TbBlock{T}[]
        degen = Int64[]
        line_nr = 0
        readline(f)
        n_wanfun = parse(Int64, readline(f))
        l = readline(f)

        while !eof(f)
            l = split(readline(f))
            if length(l)==7
                line_nr += 1
                Rtpiba = Vec3(parse(Int, l[1]), parse(Int,l[2]), parse(Int,l[3]))
                block = getfirst(x -> x.Rtpiba == Rtpiba, out)

                if block == nothing
                    block = TbBlock{T}(structure.cell' * Rtpiba, Rtpiba, Matrix{Complex{T}}(n_wanfun, n_wanfun))
                    push!(out, block)
                end
                complex = Complex{T}(parse(T, l[6]),parse(T, l[7]))/degen[div(line_nr-1,n_wanfun^2)+1]
                block.block[parse(Int, l[4]), parse(Int,l[5])] = complex
            elseif length(l)!=7
                push!(degen,parse.(Int,l)...)
            end
        end
        return out
    end
end

"""
read_dipole_file(filename::String, structure::AbstractStructure{T})

Returns and array of tuples that define the dipoles between the Wannier functions in different unit cells.
"""
function read_dipole_file(filename::String, structure::AbstractStructure{T}) where T
    open(filename) do  f
        out = DipBlock{T}[]
        readline(f)
        n_wanfun = parse(Int64, readline(f))
        while !eof(f)
            l= split(readline(f))
            ints = parse.(Int, l[1:5])
            Rtpiba = Vec3(ints[1:3]...)
            block = getfirst(x -> x.Rtpiba == Rtpiba, out)

            if block == nothing
                block = DipBlock{T}(structure.cell' * Rtpiba, Rtpiba, Matrix{Point3{T}}(n_wanfun, n_wanfun))
                push!(out, block)
            end
            dipole = Point3{T}(parse(T,l[6]),parse(T,l[8]),parse(T,l[10]))
            block.block[ints[4], ints[5]] = dipole
        end
        return out
    end
end


#stuff that is not used currently
"""
read_potential_file(filename::String, T=Float64)

Reads a Quantum Espresso potential output file.
"""
function read_potential_file(filename::String, T=Float64)
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

#-------------------------Not currently used beyond here!------------------#
function write_dip_file(filename::String,points,cell,atoms,names,direction)
    open(filename,"w") do f
        origin = points[1,1,1][1]
        write(f,["# Generated from PhD calculations\n"])
        write(f,["CRYSTAL\n","PRIMVEC\n"])
        write(f,["$(cell[1][1])  $(cell[1][2]) $(cell[1][3])\n"])
        write(f,["$(cell[2][1])  $(cell[2][2]) $(cell[2][3])\n"])
        write(f,["$(cell[3][1])  $(cell[3][2]) $(cell[3][3])\n"])
        write(f,["CONVVEC\n"])
        write(f,["$(cell[1][1])  $(cell[1][2]) $(cell[1][3])\n"])
        write(f,["$(cell[2][1])  $(cell[2][2]) $(cell[2][3])\n"])
        write(f,["$(cell[3][1])  $(cell[3][2]) $(cell[3][3])\n"])
        write(f,["PRIMCOORD\n"])
        write(f,["  $(length(atoms)) 1\n"])
        for (atom,name) in zip(atoms,names)
            write(f,["$name  $(atom.center[1])  $(atom.center[2])  $(atom.center[3])\n"])
        end

        write(f,["", "BEGIN_BLOCK_DATAGRID_3D\n", "3D_FIELD\n",
        "BEGIN_DATAGRID_3D_UNKNOWN\n"])
        write(f,"$(size(points)[1])    $(size(points)[2])     $(size(points)[3])\n")
        write(f,"$(origin[1])   $(origin[2])   $(origin[3])\n")
        write(f,"$(points[end,1,1][1][1]-origin[1])   $(points[end,1,1][1][2]-origin[2])   $(points[end,1,1][1][3]-origin[3])\n")
        write(f,"$(points[1,end,1][1][1]-origin[1])   $(points[1,end,1][1][2]-origin[2])   $(points[1,end,1][1][3]-origin[3])\n")
        write(f,"$(points[1,1,end][1][1]-origin[1])   $(points[1,1,end][1][2]-origin[2])   $(points[1,1,end][1][3]-origin[3])\n")
        for point in points
            write(f,"$(point[2][direction]) ")
        end
        write(f,"\n")
        write(f,["END_DATAGRID_3D\n", "END_BLOCK_DATAGRID_3D\n"])
    end
end

function read_xsf_file_GPU(filename::String, T=Float64)
    open(filename) do f
        while !eof(f)
            line = readline(f)
            if line == " DATAGRID_3D_DENSITY"
                line0 = split(readline(f))
                line1 = split(readline(f))
                line2 = split(readline(f))
                line3 = split(readline(f))
                line4 = split(readline(f))
                nx = parse(Int,line0[1])
                ny = parse(Int,line0[2])
                nz = parse(Int,line0[3])
                out = Complex(T,nx,ny,nz)
                line = readline(f)
                k=1
                k1=1
                k2=1
                while line!= "END_DATAGRID_3D"
                    tmp = Array{Complex{T}}(map(x->(v = tryparse(T,x); isnull(v) ? Complex(0.0,0.0) : Complex{T}(get(v),0.0)),split(line)))
                    for t in tmp
                        out[k,k1,k2] = t
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
                out = Array(Complex{T},nx,ny,nz)
                line = readline(f)
                k=1
                k1=1
                k2=1
                while line!= "END_DATAGRID_3D"
                    tmp = Array{Complex{T}}(map(x->(v = tryparse(T,x); isnull(v) ? Complex(0.0,0.0) : Complex{T}(get(v),0.0)),split(line)))
                    for t in tmp
                        out[k,k1,k2] = t
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

function write_dipole_mesh(filename,mesh::Array{Tuple{Point3{T},Point3{T}},3},direction) where T
    tmp_points = similar(mesh,WfPoint3{T})
    for (ip,p) in enumerate(mesh)
        tmp_points[ip] = WfcPoint3{T}(getfield(p[2],direction),p[1])
    end
    write_xsf_file(filename,Wfc3D(tmp_points,Point3{T}[],Atom()))
end

function write_exchanges(filename::String, structure::Structure)
    open(filename, "w") do f
        exchanges = structure.data[:exchanges]
        for (i, atom1) in enumerate(structure.atoms)
            for atom2 in structure.atoms[i + 1:end]
                J = exchange_between(atom1, atom2, exchanges)
                if J != 0
                    write(f, "$(findfirst(structure.atoms, atom1)) $(findfirst(structure.atoms, atom2)) $J\n")
                end
            end
        end
    end
end

function read_exchanges(filename::String, T=Float64)
    open(filename, "r") do f
        out = T[]
        indices = Tuple{Int,Int}[]
        max = 0
        while !eof(f)
            spl = split(readline(f))
            i, j = parse.(Int, spl[1:2])
            if j> max
                max = j
            end
            push!(indices, (i,j))
            exch = parse(T, spl[3])
            push!(out, exch)
        end
        real_out = zeros(T, (max, max))
        for (o, (i, j)) in zip(out, indices)
            real_out[i, j] = o
        end
        return real_out
    end
end
