import DFControl: searchdir, strip_split
#does this really need the checking for corruption stuff?
function read_xsf_header(::Type{T}, filename::String) where T
    open(filename) do f
        nx, ny, nz = 0, 0, 0
        origin     = zero(Point3{T})
        supercell  = zero(Mat3{T})
        read_split_parse(typ) = parse.(typ, split(readline(f)))
        lines = 0
        while true
            lines += 1
            line = readline(f)
            if line == "BEGIN_DATAGRID_3D_UNKNOWN"
                nx, ny, nz = read_split_parse(Int)
                origin     = Point3{T}(read_split_parse(T))
                t = collect(hcat([read_split_parse(T) for i=1:3]...)')::Matrix{T}
                supercell  = Mat3{T}(t)
                lines += 1 + 1 + 3
                break
            end
        end
        return nx, ny, nz, origin, supercell, lines
    end
end

function read_points_from_xsf(::Type{T}, filename::String) where {T <: AbstractFloat}
    open(filename) do f
        while !eof(f)
            line = readline(f)
            if line == " DATAGRID_3D_DENSITY" ||
               occursin("DATAGRID_3D_UNKNOWN", line)

                nx, ny, nz = parse.(Int, split(readline(f)))
                origin     = Point3{T}(parse.(T, split(readline(f))))
                a_vec      = Vec3{T}(parse.(T, split(readline(f))))
                b_vec      = Vec3{T}(parse.(T, split(readline(f))))
                c_vec      = Vec3{T}(parse.(T, split(readline(f))))
                return [origin +
                        ia * a_vec +
                        ib * b_vec +
                        ic * c_vec for ia in range(0, 1, length=nx),
                                       ib in range(0, 1, length=ny),
                                       ic in range(0, 1, length=nz)]
            end
        end
    end
end
read_points_from_xsf(filename::String) = read_points_from_xsf(Float64, filename)

"""
read_values_from_xsf(filename::String, atom::Atom, T=Float64)

Returns an Array from reading a Wannier wavefunction file.
"""
function read_values_from_xsf(::Type{T}, filename::String) where {T <: AbstractFloat}
    open(filename) do f
        while !eof(f)
            line = readline(f)
            if line == "PRIMVEC"
                cell  = [Point3{T}.(parse.(T, split(readline(f)))) for i=1:3]
            end

            if line == " DATAGRID_3D_DENSITY" || occursin("DATAGRID_3D_UNKNOWN", line)
                nx, ny, nz = parse.(Int, split(readline(f)))
	            for i = 1:4
		            readline(f)
	            end
                out     = Array{T}(undef, nx, ny, nz)
                line    = readline(f)
				counter = 1
                while line != "END_DATAGRID_3D"
                    for t in parse.(T, split(line))
	                    out[counter] = t
	                    counter += 1
                    end
                    line = readline(f)
                end
                return out
            end
        end
    end
end
read_values_from_xsf(filename::String) = read_values_from_xsf(Float64, filename)

"""
write_xsf_file(filename::String, wfc::Wfc3D{T}) where T<:AbstractFloat


Writes the real part of the Wfc3D to a .xsf file that is readable by XCrysden or VESTA.
"""
function write_xsf_file(filename::String, wfc)
    open(filename,"w") do f
        origin = wfc[1,1,1].p
        write(f,["# Generated from PhD calculations\n", "", "BEGIN_BLOCK_DATAGRID_3D\n", "3D_FIELD\n",
        "BEGIN_DATAGRID_3D_UNKNOWN\n"])
        write(f,"$(size(wfc)[1])    $(size(wfc)[2])     $(size(wfc)[3])\n")
        write(f,"$(origin[1])   $(origin[2])   $(origin[3])\n")
        write(f,"$(wfc[end,1,1].p[1]-origin[1])   $(wfc[end,1,1].p[2]-origin[2])   $(wfc[end,1,1].p[3]-origin[3])\n")
        write(f,"$(wfc[1,end,1].p[1]-origin[1])   $(wfc[1,end,1].p[2]-origin[2])   $(wfc[1,end,1].p[3]-origin[3])\n")
        write(f,"$(wfc[1,1,end].p[1]-origin[1])   $(wfc[1,1,end].p[2]-origin[2])   $(wfc[1,1,end].p[3]-origin[3])\n")
        for wfp in wfc
            write(f,"$(real(wfp.w)) ")
        end
        write(f,"\n")
        write(f,["END_DATAGRID_3D\n", "END_BLOCK_DATAGRID_3D\n"])
    end
end

#This comes from w90; it's basically a cube
const MAX_WIGNER_SEITZ_DEGENERACIES = 8 

@doc raw"""
	readhami(hami_file::AbstractString, wsvec_file::AbstractString, structure::AbstractStructure{T})

Reads `seedname_hr.dat` and `seedname_wsvec.dat` and returns a vector of TbBlocks with the hopping parameters of the Wannier Tight Binding Hamiltonian.
"""
function readhami(hami_file::AbstractString, wsvec_file::AbstractString, structure::AbstractStructure{T, LT}) where  {T<:AbstractFloat,LT<:Length{T}}
    @assert ispath(hami_file) && ispath(wsvec_file) "Please provide valid hamiltonian and wsvec files."

    wsvec_f = open(wsvec_file, "r")
    readline(wsvec_f)

    open(hami_file) do f
        out = TbBlock{T, Matrix{Complex{T}}, Matrix{Int}, Vector{Vec3{Int}}, LT}[]
        degen = Int64[]
        linenr = 0
        readline(f)
        nwanfun = parse(Int64, readline(f))
        ndegen  = parse(Int64, readline(f))
        while length(degen) < ndegen
            push!(degen, parse.(Int, split(readline(f)))...)
        end
        while !eof(f)
            l = split(readline(f))
            linenr += 1
            rpt = div(linenr - 1, nwanfun^2) + 1
            R_cryst = Vec3(parse(Int, l[1]), parse(Int, l[2]), parse(Int, l[3]))
            if length(out) < rpt
                # Reading all the wigner seitz shifts etc from the wsvec.dat file
                # TODO Performance: It's probably a better idea to have a nwanfun * nwanfun dimensional matrix with the number of degeneracies,
                #                   and a vector with all the actual shifts serialized into it.

                wigner_seitz_shifts_cryst = Vec3{Int}[]
                wigner_seitz_shifts_cart = Vec3{LT}[]
                wigner_seitz_nshift_matrix = Matrix{Int}(undef, nwanfun, nwanfun)
                n_wsvecs_read = 0
                while n_wsvecs_read < nwanfun^2
                    wanid1, wanid2 = parse.(Int, strip_split(readline(wsvec_f))[end-1:end]) #R line that should be the same as already read
                    n_ws_degeneracies = parse(Int, strip(readline(wsvec_f)))
                    t_shifts = zeros(Vec3{Int}, n_ws_degeneracies)
                    wigner_seitz_nshift_matrix[wanid1, wanid2] = n_ws_degeneracies
                    for i in 1:n_ws_degeneracies
                        t_shifts[i] = Vec3(parse.(Int, strip_split(readline(wsvec_f)))...)
                    end
                    prepend!(wigner_seitz_shifts_cryst, t_shifts)
                    prepend!(wigner_seitz_shifts_cart, (cell(structure),).*t_shifts)
                    n_wsvecs_read += 1
                    # wigner_seitz_shift_matrix[wanid1, wanid2] = t_shifts
                end

                block = TbBlock(cell(structure) * R_cryst, R_cryst, wigner_seitz_shifts_cryst, wigner_seitz_shifts_cart, wigner_seitz_nshift_matrix, degen[rpt], Matrix{Complex{T}}(I, nwanfun, nwanfun))
                push!(out, block)
            else
                block = out[rpt]
            end
            complex = Complex{T}(parse(T, l[6]), parse(T, l[7])) 
            block.block[parse(Int, l[4]), parse(Int, l[5])] = complex
        end
        return out
    end
end

#super not optimized
#TODO Test: new wigner seitz shift stuff
@doc raw"""
	read_colin_hami(upfile::AbstractString, downfile::AbstractString, up_ws_file::AbstractString, down_ws_file::AbstractString structure::AbstractStructure{T})

Returns an array of tuples that define the hopping parameters of the Wannier Tight Binding Hamiltonian.
"""
function read_colin_hami(upfile::AbstractString, downfile::AbstractString, up_ws_file::AbstractString, down_ws_file::AbstractString, structure::AbstractStructure{T}) where  T
	uphami   = readhami(upfile, up_ws_file, structure)
	downhami = readhami(downfile, down_ws_file, structure)
	dim = blocksize(uphami)
	CT = Complex{T}
	@assert dim == blocksize(downhami) "Specified files contain Hamiltonians with different dimensions of the Wannier basis."

	u1 = uphami[1]
	d1 = downhami[1]
	
	first = TbBlock(u1.R_cart, u1.R_cryst, ColinMatrix(u1.wigner_seitz_shifts, u2.wigner_seitz_shifts), u1.wigner_seitz_degeneracy, ColinMatrix(block(u1), block(d1)))
	outhami  = [first]
	for (u, d) in zip(uphami[2:end], downhami[2:end])
		tmat = ColinMatrix(block(u), block(d))
		t_shifts = ColinMatrix(u1.wigner_seitz_shifts, u2.wigner_seitz_shifts)
		degen = u.wigner_seitz_degeneracy
		push!(outhami, TbBlock(u.R_cart, u.R_cryst, t_shifts, degen, tmat))
	end
	return outhami
end

#TODO make this more robust
"""
    readhami(job::DFJob)

Goes through the job and will attempt to read the hamiltonian files.
If it finds a colinear calculation in the job it will read the up and down hamiltonians,
if the job was either nonmagnetic or noncolinear it will read only one hamiltonian file (there should be only one).
"""
function readhami(job::DFJob)

    @assert any(x -> x isa DFInput{Wannier90}, job.inputs) "No wannier90 calculations found in the job."

    seedname = getfirst(x -> x isa DFInput{Wannier90}, job.inputs).name

	hami_files  = searchdir(job.local_dir, "hr.dat")
	wsvec_files = searchdir(job.local_dir, "wsvec.dat")
	@assert !isempty(hami_files) "No hamiltonian files ($seedname_hr.dat) found."
	@assert !isempty(wsvec_files) "No wsvec files ($seedname_wsvec.dat) found."
	if !any(DFC.iscolincalc.(job.inputs))
		return readhami(joinpath(job, hami_files[1]), joinpath(job, wsvec_files[1]), job.structure)
	else
		return read_colin_hami(joinpath.((job,), hami_files)..., joinpath.((job,), wsvec_files)..., job.structure)
	end
end

"""
read_rmn_file(filename::String, structure::AbstractStructure{T})

Returns and array of tuples that define the dipoles between the Wannier functions in different unit cells.
"""
function read_rmn_file(filename::String, structure::AbstractStructure{T}) where T
    open(filename) do  f
        out = RmnBlock{T}[]
        readline(f)
        n_wanfun = parse(Int, readline(f))
        readline(f)
        while !eof(f)
            l= split(readline(f))
            R_cryst = Vec3(parse.(Int, l[1:3]))
            block = getfirst(x -> x.R_cryst == R_cryst, out)

            if block == nothing
                block = RmnBlock{T}(cell(structure)' * R_cryst, R_cryst, Matrix{Point3{T}}(I, n_wanfun, n_wanfun))
                push!(out, block)
            end
            dipole = Point3{T}(parse.(T, l[6:2:10]))
            block.block[parse.(Int, l[4:5])...] = dipole
        end
        return out
    end
end


#stuff that is not used currently
# """
# read_potential_file(filename::String, T=Float64)
#
# Reads a Quantum Espresso potential output file.
# """
# function read_potential_file(filename::String, T=Float64)
#     tmp_data=nothing
#     open(filename) do f
#         primCell = zeros(I, 3, 3)
#         while !eof(f)
#             line = readline(f)
#             if line == "PRIMVEC"
#                 for i=1:3
#                     primCell[i,:] = split(readline(f)) |> parse(T)
#                 end
#             end
#             if line == "DATAGRID_3D_UNKNOWN"
#                 nx, ny, nz = parse.(Int, split(readline(f)))
#                 for i = 1:4
#                     split(readline(f))
#                 end
#                 tmp_data=Array(T, nx, ny, nz)
#                 line = readline(f)
#                 k, k1, k2=1, 1, 1
#                 while line!= "END_DATAGRID_3D"
#                     tmp = Array{T}(map(x->(v = tryparse(T,x); isnull(v) ? 0.0 : get(v)),split(line)))
#                     for t in tmp
#                         tmp_data[k,k1,k2] = t
#                         if k<nx
#                             k+=1
#                         else
#                             k=1
#                             k1+=1
#                             if k1>ny
#                                 k1=1
#                                 k2+=1
#                                 if k2>nz
#                                     k2=1
#                                 end
#                             end
#                         end
#                     end
#                     line = readline(f)
#                 end
#                 return interpolate(tmp_data,BSpline(Cubic(Line())),OnGrid())
#             end
#         end
#     end
# end

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

# function write_dipole_mesh(filename,mesh::Array{Tuple{Point3{T},Point3{T}},3},direction) where T
#     tmp_points = similar(mesh,WfPoint3{T})
#     for (ip,p) in enumerate(mesh)
#         tmp_points[ip] = WfcPoint3{T}(getfield(p[2],direction),p[1])
#     end
#     write_xsf_file(filename,Wfc3D(tmp_points,Point3{T}[],Atom()))
# end

function write_exchanges(filename::String, structure::Structure)
    open(filename, "w") do f
        exchanges = data(structure)[:exchanges]
        for (i, atom1) in enumerate(atoms(structure))
            for (i2, atom2) in enumerate(atoms(structure)[i + 1:end])
                J = exchange_between(atom1, atom2, exchanges)
                if J != 0
                    write(f, "$i:$(name(atom1)) -> $(i2+i):$(name(atom2)) $J\n")
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

function read_chk(filename)
    f = FortranFile(filename)
    header = String(read(f, FString{33}))
    n_bands = Int(read(f, Int32))
    n_excluded_bands = Int(read(f, Int32))
    # exclude_bands_t = zeros(Int32, n_excluded_bands)
    read(f, (Int32, n_excluded_bands))
    # exclude_bands= convert.(Int, exclude_bands_t)
    read(f)
    real_lattice = Mat3(read(f, (Float64, 3, 3))...)
    recip_lattice = K_CART_TYPE{Float64}.(Mat3(read(f, (Float64, 3, 3))...)')
    n_kpoints = read(f, Int32)
    mp_grid = Vec3(Int.(read(f, (Int32, 3)))...)
    kpt_lattice_t = read(f, (Float64, 3, n_kpoints))
    kpt_lattice = [Vec3(kpt_lattice_t[:, i]...) for i = 1:size(kpt_lattice_t)[2]]
    k_nearest_neighbors = read(f, Int32)

    n_wann = read(f, Int32)
    chkpt = strip(String(read(f, FString{20})))
    have_disentangled = read(f, Int32) == 1 ? true : false
    if have_disentangled
        omega_invariant = read(f, Float64)
        lwindow = map(x-> x==1 ? true : false, read(f, (Int32, n_bands, n_kpoints)))
        ndimwin = read(f, (Int32, n_kpoints))
        U_matrix_opt = read(f, (Complex{Float64}, n_bands, n_wann, n_kpoints))
    else
        omega_invariant = 0.0
        lwindow = fill(true, 1, n_kpoints)
        ndimwin = fill(n_wann, n_kpoints)
        U_matrix_opt= Array{Complex{Float64}, 3}()
    end
    U_matrix = read(f, (Complex{Float64}, n_wann, n_wann, n_kpoints))

    # Combined effect of disentanglement and localization
    V_matrix = Array{Complex{Float64},3}(undef, n_bands, n_wann, n_kpoints)
    if have_disentangled
        for ik in 1:n_kpoints
            V_matrix[:, :, ik] = U_matrix_opt[:, :, ik] * U_matrix[:, :, ik]
        end
    else
        V_matrix = U_matrix
    end

    m_matrix = read(f, (Complex{Float64}, n_wann, n_wann, k_nearest_neighbors, n_kpoints))
    wannier_centers_t = read(f, (Float64, 3, n_wann))
    wannier_centers = [Point3(wannier_centers_t[:, i]...) for i = 1:size(wannier_centers_t)[2]]
    wannier_spreads = read(f, (Float64, n_wann))

    wb = read(f, (Float64, k_nearest_neighbors)) #this requires patched w90

    return (
        n_bands=n_bands,
        n_excluded_bands=n_excluded_bands,
        cell=real_lattice,
        recip_cell=recip_lattice,
        n_kpoints=n_kpoints,
        mp_grid=mp_grid,
        kpoints=kpt_lattice,
        n_nearest_neighbors=k_nearest_neighbors,
        neighbor_weights=wb,
        n_wann=n_wann,
        have_disentangled=have_disentangled,
        Ω_invariant = omega_invariant,
        lwindow=lwindow,
        ndimwin =ndimwin,
        U_matrix_opt=U_matrix_opt,
        U_matrix=U_matrix,
        V_matrix=V_matrix,
        m_matrix=m_matrix,
        wannier_centers=wannier_centers,
        wannier_spreads=wannier_spreads
    )
end

function read_eig(filename)
    t = readdlm(filename)
    n_bands = maximum(t[:,1])
    n_kpoints = maximum(t[:,2])

    #we follow order of w90 eigval matrix
    Hk = Matrix{Float64}(undef, Int(n_bands), Int(n_kpoints))
    for x in 1:size(t)[1]
        tv = t[x, :]
        Hk[Int(tv[1]), Int(tv[2])] = tv[3]
    end
    return Hk
end

function read_unk_nc(file)
    f = FortranFile(file)
    ngx, ngy, ngz, nk, nbnd = read(f, (Int32, 5))
    Uk = zeros(ComplexF64, ngx, ngy, ngz,142, 2)
    for i in 1:nbnd
        Uk[:,:,:, i, 1] .= read(f, (ComplexF64, 45,45, 36))
        Uk[:,:,:,i, 2]  .= read(f, (ComplexF64, 45,45, 36))
    end
    return Uk
end

mutable struct ReciprocalOverlaps{T}
    k_id           ::Int
    neighbor_ids   ::Vector{Int}
    b_vectors_cryst::Vector{Vec3{Int}} #connecting vector in crystalline coordinates
    overlaps       ::Vector{Matrix{Complex{T}}}
end

function ReciprocalOverlaps{T}(k_id::Int, n_nearest_neighbors::Int, nbands::Int) where {T}
    overlap_matrices = [Matrix{Complex{T}}(undef, nbands, nbands) for i=1:n_nearest_neighbors]
    ReciprocalOverlaps{T}(k_id, zeros(Int, n_nearest_neighbors), zeros(Vec3{Int}, n_nearest_neighbors), overlap_matrices)
end

function fill_overlaps!(grid::Vector{AbInitioKPoint{T}}, mmn_filename::AbstractString, uHu_filename::AbstractString, wannier_chk_params) where {T}
    num_wann = wannier_chk_params.n_wann
    uHu_file = FortranFile(uHu_filename)
    read(uHu_file, FString{20})
    read(uHu_file, (Int32, 3))

    open(mmn_filename, "r") do f
        readline(f) #header
        nbands, nkpoints, n_nearest_neighbors = parse.(Int, strip_split(readline(f)))
        #pre setup uHu
        for k in grid
            k.uHu = [Matrix{Complex{T}}(undef, num_wann, num_wann) for m=1:n_nearest_neighbors, n=1:n_nearest_neighbors]
        end
        neighbor_counter = 1
        for i in 1:nkpoints*n_nearest_neighbors
            sline = strip_split(readline(f))
            cur_neighbor = mod1(neighbor_counter, n_nearest_neighbors) 

            ik, ik2 = parse.(Int, sline[1:2])

            overlap_ab_initio_gauge = Matrix{Complex{T}}(undef, nbands, nbands)
            for n in eachindex(overlap_ab_initio_gauge)
                overlap_ab_initio_gauge[n] = complex(parse.(T, strip_split(readline(f)))...)
            end

            vmat_ik = wannier_chk_params.V_matrix[:, :, ik]
            vmat_ik2 = wannier_chk_params.V_matrix[:, :, ik2]
            first_band_id_ik = findfirst(wannier_chk_params.lwindow[:, ik])
            first_band_id_ik2 = findfirst(wannier_chk_params.lwindow[:, ik2])

            num_states_ik = wannier_chk_params.ndimwin[ik]
            num_states_ik2 = wannier_chk_params.ndimwin[ik2]

            V1 = vmat_ik[1:num_states_ik, 1:num_wann]
            V2 = vmat_ik2[1:num_states_ik2, 1:num_wann]


            disentanglement_range_k1 = first_band_id_ik:first_band_id_ik+num_states_ik-1
            disentanglement_range_k2 = first_band_id_ik2:first_band_id_ik2+num_states_ik2-1
            S12 = overlap_ab_initio_gauge[disentanglement_range_k1, disentanglement_range_k2]

            kpoint = grid[ik]

            vr = (wannier_chk_params.recip_cell * parse(Vec3{Int}, sline[3:5]) + grid[ik2].k_cart) - kpoint.k_cart

            V1_T = V1'
            S12_V2 = S12 * V2
            kpoint.overlaps[cur_neighbor] =  V1_T * S12_V2
            k_eigvals_mat = diagm(kpoint.eigvals[disentanglement_range_k1])
            kpoint.hamis[cur_neighbor] = V1_T * k_eigvals_mat * S12_V2
            neighbor_counter += 1
            for nearest_neighbor2 in 1:n_nearest_neighbors
                ik3 = kpoint.neighbors[nearest_neighbor2].k_id2
                first_band_id_ik3 = findfirst(wannier_chk_params.lwindow[:, ik3])
                num_states_ik3    = wannier_chk_params.ndimwin[ik3]

                V3 = wannier_chk_params.V_matrix[1:num_states_ik3, 1:num_wann, ik3]
                

                uHu_k2_k3 = transpose(read(uHu_file, (ComplexF64, nbands, nbands)))
                disentanglement_range_k3 = first_band_id_ik3:first_band_id_ik3+num_states_ik3-1
                kpoint.uHu[nearest_neighbor2, cur_neighbor] =  V3' * uHu_k2_k3[disentanglement_range_k3, disentanglement_range_k2] * V2
            end
        end
        return grid
    end
end

function fill_k_neighbors!(kpoints::Vector{AbInitioKPoint{T}}, file::AbstractString, recip_cell::Mat3) where {T}
    open(file, "r") do f
        line = readline(f)
        while line != "begin kpoints"
            line = readline(f)
        end
        nkpoints = parse(Int, strip(readline(f)))
        @assert nkpoints == length(kpoints) "Number kpoints in seedname.nnkp doesn't match with the number of kpoints in seedname.chk."

        while line != "begin nnkpts"
            line = readline(f)
        end
        n_nearest_neighbors = parse(Int, strip(readline(f)))


        counter = 1
        line = readline(f)
        while line != "end nnkpts"
            counter = mod1(counter, n_nearest_neighbors)
            sline = strip_split(line)
            ik, ik2 = parse.(Int, sline[1:2])

            vr = (recip_cell * parse(Vec3{Int}, sline[3:5]) + kpoints[ik2].k_cart) - kpoints[ik].k_cart
            push!(kpoints[ik].neighbors, KBond(ik, ik2, vr))
            push!(kpoints[ik].overlaps, Matrix{Complex{T}}(undef, 0, 0))
            push!(kpoints[ik].hamis, Matrix{Complex{T}}(undef, 0, 0))
            kpoints[ik].uHu = Matrix{Matrix{Complex{T}}}(undef, n_nearest_neighbors, n_nearest_neighbors)
            counter += 1
            line = readline(f)
        end
        return kpoints
    end
end

# function read_eig(file::AbstractString, n_bands::Int, n_kpoints::Int)
#     eigvals = Matrix{Float64}(undef, n_bands, n_kpoints)
#     open(f, "r") do f
#         while !eof(f)
#             ik, val = parse.((Int, Float64), strip_split(readline(f)))
#             eigvals[ib, ik] = val
#         end
#     end
#     return eigvals
# end


