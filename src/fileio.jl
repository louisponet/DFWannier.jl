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
function write_xsf_file(filename::String, wfc, structure; value_func=x -> norm(x))
    open(filename,"w") do f
        origin = wfc.points[1,1,1]
        write(f,"# Generated from PhD calculations\n")
        write(f, "CRYSTAL\n")
        c = ustrip.(cell(structure)')
        write(f, "PRIMVEC\n")
        write(f, "$(c[1,1]) $(c[1,2]) $(c[1,3])\n")
        write(f, "$(c[2,1]) $(c[2,2]) $(c[2,3])\n")
        write(f, "$(c[3,1]) $(c[3,2]) $(c[3,3])\n")
        write(f, "CONVVEC\n")
        write(f, "$(c[1,1]) $(c[1,2]) $(c[1,3])\n")
        write(f, "$(c[2,1]) $(c[2,2]) $(c[2,3])\n")
        write(f, "$(c[3,1]) $(c[3,2]) $(c[3,3])\n")
        write(f, "PRIMCOORD\n")
        write(f, "$(length(atoms(structure))) 1\n")
        for at in atoms(structure)
            n = element(at).symbol
            p = ustrip.(position_cart(at))
            write(f, "$n $(p[1]) $(p[2]) $(p[3])\n")
        end
        write.((f,),["", "BEGIN_BLOCK_DATAGRID_3D\n", "3D_FIELD\n", "BEGIN_DATAGRID_3D_UNKNOWN\n"])
        write(f,"$(size(wfc)[1])    $(size(wfc)[2])     $(size(wfc)[3])\n")
        write(f,"$(origin[1])   $(origin[2])   $(origin[3])\n")
        write(f,"$(wfc.points[end,1,1][1]-origin[1])   $(wfc.points[end,1,1][2]-origin[2])   $(wfc.points[end,1,1][3]-origin[3])\n")
        write(f,"$(wfc.points[1,end,1][1]-origin[1])   $(wfc.points[1,end,1][2]-origin[2])   $(wfc.points[1,end,1][3]-origin[3])\n")
        write(f,"$(wfc.points[1,1,end][1]-origin[1])   $(wfc.points[1,1,end][2]-origin[2])   $(wfc.points[1,1,end][3]-origin[3])\n")
        for wfp in wfc.values
            write(f,"$(value_func(wfp)) ")
        end
        write(f,"\n")
        write.((f,), ["END_DATAGRID_3D\n", "END_BLOCK_DATAGRID_3D\n"])
    end
end

#This comes from w90; it's basically a cube
const MAX_WIGNER_SEITZ_DEGENERACIES = 8

function read_wsvec(file, nwanfun::Integer)
    out = NamedTuple{(:R_cryst, :shifts_cryst, :nshifts),
                     Tuple{Vec3{Int}, Matrix{Vector{Vec3{Int}}}, Matrix{Int}}}[]
 
    open(file, "r") do f
        readline(f)
        n_wsvec_read = 0
        while !eof(f)
            n_wsvec_read = 0
            R_cryst = zero(Vec3{Int})
            shifts_cryst = [Vec3{Int}[] for i=1:nwanfun, j=1:nwanfun]
            nshifts = [0 for i=1:nwanfun, j=1:nwanfun]
            while n_wsvec_read < nwanfun^2
                l = strip_split(readline(f))
                R_cryst = iszero(R_cryst) ? parse(Vec3{Int}, l[1:3]) : R_cryst
                i, j = parse.(Int, l[4:5])
                nshifts[i, j] = parse(Int, strip(readline(f)))
                for ip = 1:nshifts[i, j]
                    push!(shifts_cryst[i, j], parse(Vec3{Int}, strip_split(readline(f))))
                end
                n_wsvec_read += 1
            end
            push!(out, (R_cryst=R_cryst, shifts_cryst=shifts_cryst, nshifts=nshifts))
        end
    end
    return out
end 

function readhami(chk_file::AbstractString, eig_file::AbstractString)
    chk = read_chk(chk_file)
    ws_shifts, ws_nshifts = generate_wsvec(chk)
    v_mat = chk.V_matrix
    eigvals = read_eig(eig_file)
    R_cryst, degens = wigner_seitz_points(chk)
    Hq = map(1:length(chk.kpoints)) do ik
        v = v_mat[1:num_states(chk, ik), 1:chk.n_wann, ik]

        return v' * diagm(eigvals[disentanglement_range(chk, ik), ik]) * v
    end
    c = chk.cell'
    LT = eltype(c)
    T = eltype(eigvals)
    out = [TbBlock{T, Matrix{Complex{T}}, Matrix{Int}, Matrix{Vector{Vec3{Int}}}, LT, Matrix{Vector{Vec3{LT}}}}(c*R, R, shifts, map(x->map(y->c*y,x), shifts), nshifts, d, zeros(ComplexF64, chk.n_wann, chk.n_wann), zeros(ComplexF64, chk.n_wann, chk.n_wann)) for (R, shifts, nshifts, d) in zip(R_cryst, ws_shifts, ws_nshifts, degens)]
    fourier_q_to_R(chk.kpoints, R_cryst) do iR, ik, phase
        @inbounds out[iR].block .+= phase .* Hq[ik]
    end
    for o in out
        o.block ./= length(chk.kpoints)
    end

    for (iR, h) in enumerate(out)
        for i in eachindex(h.block)
            nshifts = h.wigner_seitz_nshifts[i]
            for is = 1:nshifts
                rcryst = h.R_cryst + h.wigner_seitz_shifts_cryst[i][is]
                h1 = out[rcryst]
                if h1 === nothing
                    h1 = TbBlock{T, Matrix{Complex{T}}, Matrix{Int}, Matrix{Vector{Vec3{Int}}}, LT, Matrix{Vector{Vec3{LT}}}}(c*rcryst, rcryst, fill(Vec3{Int}[], size(h.block)),fill(Vec3{LT}[], size(h.block)), zeros(Int, chk.n_wann, chk.n_wann), 1, zeros(ComplexF64, chk.n_wann, chk.n_wann), zeros(ComplexF64, chk.n_wann, chk.n_wann))
                    push!(out, h1)
                end
                h1.preprocessed_block[i] += h.block[i] / (h.wigner_seitz_degeneracy * nshifts)
            end
        end
    end
    return out
end

@doc raw"""
	readhami(hami_file::AbstractString, wsvec_file::AbstractString, structure::AbstractStructure{T})

Reads `seedname_hr.dat` and `seedname_wsvec.dat` and returns a vector of TbBlocks with the hopping parameters of the Wannier Tight Binding Hamiltonian.
"""
function readhami(hami_file::AbstractString, wsvec_file::AbstractString, structure::AbstractStructure{T, LT}) where  {T<:AbstractFloat,LT<:Length{T}}
    @assert ispath(hami_file) && ispath(wsvec_file) "Please provide valid hamiltonian and wsvec files."

    wsvec_f = open(wsvec_file, "r")
    readline(wsvec_f)

    out = TbBlock{T, Matrix{Complex{T}}, Matrix{Int}, Matrix{Vector{Vec3{Int}}}, LT, Matrix{Vector{Vec3{LT}}}}[]
    # out = TbBlock{T, Matrix{Complex{T}}, Matrix{Int}, Matrix{Vector{Vec3{Int}}}, LT}[]
    degen = Int64[]
    nwanfun = 0
    ndegen  = 0
    open(hami_file) do f
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

                # wigner_seitz_shifts_cryst = Vec3{Int}[]
                wigner_seitz_shifts_cryst = Matrix{Vector{Vec3{Int}}}(undef, nwanfun, nwanfun)
                # wigner_seitz_shifts_cart = Vec3{LT}[]
                wigner_seitz_shifts_cart = Matrix{Vector{Vec3{LT}}}(undef, nwanfun, nwanfun)
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
                    wigner_seitz_shifts_cryst[wanid1, wanid2] = t_shifts
                    wigner_seitz_shifts_cart[wanid1, wanid2] = (cell(structure),).*t_shifts
                    # prepend!(wigner_seitz_shifts_cryst, t_shifts)
                    # prepend!(wigner_seitz_shifts_cart, (cell(structure),).*t_shifts)
                    n_wsvecs_read += 1
                    # wigner_seitz_shift_matrix[wanid1, wanid2] = t_shifts
                end

                block = TbBlock(cell(structure) * R_cryst, R_cryst, wigner_seitz_shifts_cryst, wigner_seitz_shifts_cart, wigner_seitz_nshift_matrix, degen[rpt], Matrix{Complex{T}}(I, nwanfun, nwanfun))
                # block = TbBlock(cell(structure) * R_cryst, R_cryst, wigner_seitz_shifts_cryst, wigner_seitz_nshift_matrix, degen[rpt], Matrix{Complex{T}}(I, nwanfun, nwanfun))
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
	
	first = TbBlock(u1.R_cart,
	                u1.R_cryst,
	                ColinMatrix(u1.wigner_seitz_shifts_cryst, d1.wigner_seitz_shifts_cryst),
	                ColinMatrix(u1.wigner_seitz_shifts_cart, d1.wigner_seitz_shifts_cart),
	                ColinMatrix(u1.wigner_seitz_nshifts, d1.wigner_seitz_nshifts),
	                u1.wigner_seitz_degeneracy,
	                ColinMatrix(block(u1), block(d1)))

	outhami  = [first]
	for (u, d) in zip(uphami[2:end], downhami[2:end])
		tmat = ColinMatrix(block(u), block(d))
		t_shifts_cryst = ColinMatrix(u.wigner_seitz_shifts_cryst, d.wigner_seitz_shifts_cryst)
		t_shifts_cart = ColinMatrix(u.wigner_seitz_shifts_cart, d.wigner_seitz_shifts_cart)
		t_nshifts = ColinMatrix(u.wigner_seitz_nshifts, d.wigner_seitz_nshifts)
		degen = u.wigner_seitz_degeneracy
		push!(outhami, TbBlock(u.R_cart, u.R_cryst, t_shifts_cryst,t_shifts_cart,t_nshifts, degen, tmat))
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
    jld_file = joinpath(job, "hami.jld2")
    if ispath(jld_file)
        return DFC.load(jld_file)["hami"]
    end
	hami_files  = searchdir(job.local_dir, "hr.dat")
	wsvec_files = searchdir(job.local_dir, "wsvec.dat")
	@assert !isempty(hami_files) "No hamiltonian files ($(seedname)_hr.dat) found."
	@assert !isempty(wsvec_files) "No wsvec files ($(seedname)_wsvec.dat) found."
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

# function read_chk(filename)
#     # try
#         f = FortranFile(filename)
#         header = String(read(f, FString{33}))
#         n_bands = Int(read(f, Int32))
#         n_excluded_bands = Int(read(f, Int32))
#         # exclude_bands_t = zeros(Int32, n_excluded_bands)
#         read(f, (Int32, n_excluded_bands))
#         # exclude_bands= convert.(Int, exclude_bands_t)
#         read(f)
#         real_lattice = Mat3(read(f, (Float64, 3, 3))...)
#         recip_lattice = K_CART_TYPE{Float64}.(Mat3(read(f, (Float64, 3, 3))...)')
#         n_kpoints = read(f, Int32)
#         mp_grid = Vec3(Int.(read(f, (Int32, 3)))...)
#         kpt_lattice_t = read(f, (Float64, 3, n_kpoints))
#         kpt_lattice = [Vec3(kpt_lattice_t[:, i]...) for i = 1:size(kpt_lattice_t)[2]]
#         k_nearest_neighbors = read(f, Int32)

#         n_wann = read(f, Int32)
#         chkpt = strip(String(read(f, FString{20})))
#         have_disentangled = read(f, Int32) == 1 ? true : false
#         if have_disentangled
#             omega_invariant = read(f, Float64)
#             lwindow = map(x-> x==1 ? true : false, read(f, (Int32, n_bands, n_kpoints)))
#             ndimwin = read(f, (Int32, n_kpoints))
#             U_matrix_opt = read(f, (Complex{Float64}, n_bands, n_wann, n_kpoints))
#         else
#             omega_invariant = 0.0
#             lwindow = fill(true, 1, n_kpoints)
#             ndimwin = fill(n_wann, n_kpoints)
#             U_matrix_opt= Array{Complex{Float64}, 3}()
#         end
#         U_matrix = permutedims(read(f, (Complex{Float64}, n_wann, n_wann, n_kpoints)),(2,1,3))

#         # Combined effect of disentanglement and localization
#         V_matrix = Array{Complex{Float64},3}(undef, n_bands, n_wann, n_kpoints)
#         if have_disentangled
#             for ik in 1:n_kpoints
#                 V_matrix[:, :, ik] = U_matrix_opt[:, :, ik] * U_matrix[:, :, ik]
#             end
#         else
#             V_matrix = U_matrix
#         end

#         m_matrix = read(f, (Complex{Float64}, n_wann, n_wann, k_nearest_neighbors, n_kpoints))
#         wannier_centers_t = read(f, (Float64, 3, n_wann))
#         wannier_centers = [Point3(wannier_centers_t[:, i]...) for i = 1:size(wannier_centers_t)[2]]
#         wannier_spreads = read(f, (Float64, n_wann))

#         wb = read(f, (Float64, k_nearest_neighbors)) #this requires patched w90

#         return (
#             n_bands=n_bands,
#             n_excluded_bands=n_excluded_bands,
#             cell=real_lattice',
#             recip_cell=recip_lattice,
#             n_kpoints=n_kpoints,
#             mp_grid=mp_grid,
#             kpoints=kpt_lattice,
#             n_nearest_neighbors=k_nearest_neighbors,
#             neighbor_weights=wb,
#             n_wann=n_wann,
#             have_disentangled=have_disentangled,
#             Ω_invariant = omega_invariant,
#             lwindow=lwindow,
#             ndimwin =ndimwin,
#             U_matrix_opt=U_matrix_opt,
#             U_matrix=U_matrix,
#             V_matrix=V_matrix,
#             m_matrix=m_matrix,
#             wannier_centers=wannier_centers,
#             wannier_spreads=wannier_spreads
#         )
    # catch
    #     open(filename, "r") do f
    #         header = readline(f)
    #         n_bands = parse(Int, readline(f))
    #         n_excluded_bands = parse(Int,readline(f))
    #         # exclude_bands_t = zeros(Int32, n_excluded_bands)
    #         for i=1:n_excluded_bands
    #             readline(f)
    #         end
    #         # exclude_bands= convert.(Int, exclude_bands_t)
    #         # read(f)
    #         ssl = () -> strip_split(readline(f))
    #         real_lattice = Mat3(parse.(Float64, ssl()))
    #         recip_lattice = K_CART_TYPE{Float64}.(parse.(Float64, ssl())')
    #         n_kpoints = parse(Int, readline(f))
    #         mp_grid = parse(Vec3{Int}, ssl())
    #         kpt_lattice = [zero(Vec3{Float64}) for i=1:n_kpoints]
    #         for i = 1:n_kpoints
    #             kpt_lattice[i] = parse(Vec3{Float64}, ssl())
    #         end
    #         k_nearest_neighbors = parse(Int, readline(f))

    #         n_wann = parse(Int, readline(f))
    #         chkpt = readline(f)
    #         have_disentangled = parse(Bool, readline(f))
    #         if have_disentangled
    #             omega_invariant = parse(Float64, readline(f))
    #             lwindow = [parse(Bool, readline(f)) for i=1:n_bands, j=1:n_kpoints]
    #             # lwindow = map(x-> x == 1 ? true : false, )
    #             ndimwin = [parse(Int, readline(f)) for i=1:n_kpoints]
    #             U_matrix_opt = [complex(parse.(Float64, ssl())...) for i=1:n_bands, j=1:n_wann, k=1:n_kpoints]
    #         else
    #             omega_invariant = 0.0
    #             lwindow = fill(true, 1, n_kpoints)
    #             ndimwin = fill(n_wann, n_kpoints)
    #             U_matrix_opt= Array{Complex{Float64}, 3}()
    #         end
    #         U_matrix = permutedims([complex(parse.(Float64, ssl())...) for i=1:n_wann, j=1:n_wann, k=1:n_kpoints],(2,1,3))

    #         # Combined effect of disentanglement and localization
    #         V_matrix = Array{Complex{Float64},3}(undef, n_bands, n_wann, n_kpoints)
    #         if have_disentangled
    #             for ik in 1:n_kpoints
    #                 V_matrix[:, :, ik] = U_matrix_opt[:, :, ik] * U_matrix[:, :, ik]
    #             end
    #         else
    #             V_matrix = U_matrix
    #         end

    #         m_matrix = [complex(parse.(Float64, ssl())...) for i=1:n_wann, j=1:n_wann, k=1:k_nearest_neighbors, l=1:n_kpoints]
    #         wannier_centers = [parse(Point3{Float64}, ssl()) for i=1:n_wann]
    #         wannier_spreads = [parse(Float64, readline(f)) for i=1:n_wann]

    #         # wb = [parse(Float64, readline(f)) for i=1:k_nearest_neighbors] #this requires patched w90

    #         return (
    #             n_bands=n_bands,
    #             n_excluded_bands=n_excluded_bands,
    #             cell=real_lattice',
    #             recip_cell=recip_lattice,
    #             n_kpoints=n_kpoints,
    #             mp_grid=mp_grid,
    #             kpoints=kpt_lattice,
    #             n_nearest_neighbors=k_nearest_neighbors,
    #             neighbor_weights=wb,
    #             n_wann=n_wann,
    #             have_disentangled=have_disentangled,
    #             Ω_invariant = omega_invariant,
    #             lwindow=lwindow,
    #             ndimwin =ndimwin,
    #             U_matrix_opt=U_matrix_opt,
    #             U_matrix=U_matrix,
    #             V_matrix=V_matrix,
    #             m_matrix=m_matrix,
    #             wannier_centers=wannier_centers,
    #             wannier_spreads=wannier_spreads
    #         )
    #     end
    #     # return read_chk_formatted(filename)
    # end
# end
#TODO: cleanup
const CART_TYPE{T} = Quantity{T,Unitful.𝐋,Unitful.FreeUnits{(Ang,),Unitful.𝐋,nothing}} 
function read_chk(filename)
    f = FortranFile(filename)
    header = String(read(f, FString{33}))
    n_bands = Int(read(f, Int32))
    n_excluded_bands = Int(read(f, Int32))
    # exclude_bands_t = zeros(Int32, n_excluded_bands)
    read(f, (Int32, n_excluded_bands))
    # exclude_bands= convert.(Int, exclude_bands_t)
    read(f)
    real_lattice = CART_TYPE{Float64}.(Mat3(read(f, (Float64, 3, 3))...))
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


const WS_DISTANCE_TOL = 1e-5
function generate_wsvec(chk)
    R_cryst, degens = wigner_seitz_points(chk)
    wannier_centers = chk.wannier_centers
    ws_shifts_cryst = [[Vec3{Int}[zero(Vec3{Int})] for i=1:chk.n_wann, j=1:chk.n_wann] for iR = 1:length(R_cryst)]
    ws_nshifts      = [zeros(Int, chk.n_wann,chk.n_wann) for iR = 1:length(R_cryst)]
    c = ustrip.(chk.cell')
    ic = inv(c)
    for (iR, R) in enumerate(R_cryst)
        r_cart = c * R
        for i in 1:chk.n_wann, j=1:chk.n_wann
            best_r_cart = -wannier_centers[i] + r_cart + wannier_centers[j]
            nr = norm(best_r_cart)

            r_cryst = ic * best_r_cart

            for l in -3:3, m in -3:3, n in -3:3
                lmn = Vec3(l, m, n)
                test_r_cryst = r_cryst + lmn .* chk.mp_grid
                test_r_cart  = c * test_r_cryst
                if norm(test_r_cart) < nr
                    best_r_cart = test_r_cart
                    nr = norm(test_r_cart)
                    ws_shifts_cryst[iR][i, j][1] = lmn .* chk.mp_grid
                end
            end

            if nr < WS_DISTANCE_TOL
                ws_nshifts[iR][i, j] = 1
                ws_shifts_cryst[iR][i, j][1] = Vec3(0, 0, 0)
            else
                best_r_cryst = ic * best_r_cart
                orig_shift = ws_shifts_cryst[iR][i,j][1]
                for l in -3:3, m in -3:3, n in -3:3
                    lmn = Vec3(l, m, n)
                    test_r_cryst = best_r_cryst + lmn .* chk.mp_grid
                    test_r_cart  = c * test_r_cryst
                    if abs(norm(test_r_cart) - nr) < WS_DISTANCE_TOL
                        ws_nshifts[iR][i, j] += 1
                        if ws_nshifts[iR][i, j] == 1
                            ws_shifts_cryst[iR][i, j][ws_nshifts[iR][i, j]] = orig_shift + lmn .* chk.mp_grid
                        else
                            push!(ws_shifts_cryst[iR][i, j], orig_shift + lmn .* chk.mp_grid)
                        end
                    end
                end
            end
        end
    end
    return ws_shifts_cryst, ws_nshifts
end





#TODO: speedup, we don't need all of them if things were disentangled

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

function read_uHu(file)
    try 
        uHu_file = FortranFile(file)
        read(uHu_file, FString{20})
        nbands, nkpoints, n_nearest_neighbors = read(uHu_file, (Int32, 3))
        out = [Matrix{ComplexF64}(undef, nbands, nbands) for i=1:nkpoints*n_nearest_neighbors^2]
        for i=1:nkpoints*n_nearest_neighbors^2
            out[i] = transpose(read(uHu_file, (ComplexF64, nbands, nbands)))
        end
        return out
    catch
        open(file, "r") do f
            readline(f)
            nbands, nkpoints, n_nearest_neighbors = parse.(Int, strip_split(readline(f)))
            out = [Matrix{ComplexF64}(undef, nbands, nbands) for i=1:nkpoints*n_nearest_neighbors^2]
            for i=1:nkpoints*n_nearest_neighbors^2
                for j = 1:nbands
                    for k = 1:nbands
                        out[i][j,k] = complex(parse.(Float64, strip_split(readline(f)))...)
                    end
                end
            end
            return out
        end
    end
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

# function fill_overlaps!(grid::Vector{AbInitioKPoint{T}}, mmn_filename::AbstractString, uHu_filename::AbstractString, wannier_chk_params) where {T}
#     num_wann = wannier_chk_params.n_wann
#     uHu_file = FortranFile(uHu_filename)
#     read(uHu_file, FString{20})
#     nbands, nkpoints, n_nearest_neighbors = read(uHu_file, (Int32, 3))

#     open(mmn_filename, "r") do f
#         readline(f) #header
#         nbands, nkpoints, n_nearest_neighbors = parse.(Int, strip_split(readline(f)))
#         #pre setup uHu
#         for k in grid
#             k.uHu = [Matrix{Complex{T}}(undef, num_wann, num_wann) for m=1:n_nearest_neighbors, n=1:n_nearest_neighbors]
#         end
#         neighbor_counter = 1
#         for i in 1:nkpoints*n_nearest_neighbors
#             sline = strip_split(readline(f))
#             cur_neighbor = mod1(neighbor_counter, n_nearest_neighbors) 

#             ik, ik2 = parse.(Int, sline[1:2])

#             overlap_ab_initio_gauge = Matrix{Complex{T}}(undef, nbands, nbands)
#             for n in eachindex(overlap_ab_initio_gauge)
#                 overlap_ab_initio_gauge[n] = complex(parse.(T, strip_split(readline(f)))...)
#             end

#             vmat_ik = wannier_chk_params.V_matrix[:, :, ik]
#             vmat_ik2 = wannier_chk_params.V_matrix[:, :, ik2]
#             first_band_id_ik = findfirst(wannier_chk_params.lwindow[:, ik])
#             first_band_id_ik2 = findfirst(wannier_chk_params.lwindow[:, ik2])

#             num_states_ik = wannier_chk_params.ndimwin[ik]
#             num_states_ik2 = wannier_chk_params.ndimwin[ik2]

#             V1 = vmat_ik[1:num_states_ik, 1:num_wann]
#             V2 = vmat_ik2[1:num_states_ik2, 1:num_wann]


#             disentanglement_range_k1 = first_band_id_ik:first_band_id_ik+num_states_ik-1
#             disentanglement_range_k2 = first_band_id_ik2:first_band_id_ik2+num_states_ik2-1
#             S12 = overlap_ab_initio_gauge[disentanglement_range_k1, disentanglement_range_k2]

#             kpoint = grid[ik]

#             vr = (wannier_chk_params.recip_cell * parse(Vec3{Int}, sline[3:5]) + grid[ik2].k_cart) - kpoint.k_cart

#             V1_T = V1'
#             S12_V2 = S12 * V2
#             kpoint.overlaps[cur_neighbor] =  V1_T * S12_V2
#             k_eigvals_mat = diagm(kpoint.eigvals[disentanglement_range_k1])
#             kpoint.hamis[cur_neighbor] = V1_T * k_eigvals_mat * S12_V2
#             neighbor_counter += 1
#             for nearest_neighbor2 in 1:n_nearest_neighbors
#                 ik3 = kpoint.neighbors[nearest_neighbor2].k_id2
#                 first_band_id_ik3 = findfirst(wannier_chk_params.lwindow[:, ik3])
#                 num_states_ik3    = wannier_chk_params.ndimwin[ik3]

#                 V3 = wannier_chk_params.V_matrix[1:num_states_ik3, 1:num_wann, ik3]
                

#                 uHu_k2_k3 = transpose(read(uHu_file, (ComplexF64, nbands, nbands)))
#                 disentanglement_range_k3 = first_band_id_ik3:first_band_id_ik3+num_states_ik3-1
#                 kpoint.uHu[nearest_neighbor2, cur_neighbor] =  V3' * uHu_k2_k3[disentanglement_range_k3, disentanglement_range_k2] * V2
#             end
#         end
#         return grid
#     end
# end

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

function read_wannier_functions(job)
    wancalc_ids = findall(x -> DFC.package(x) == Wannier90, DFC.inputs(job))
    wanfuncs = Vector{WannierFunction}[]
    for wancalc in DFC.inputs(job)[wancalc_ids]
        t_funcs = WannierFunction[]
        # here we assume that if it's a spinor calculation, and all the upre, upim, downre, downim are there
        if DFC.hasflag(wancalc, :spinors) && wancalc[:spinors]
            upre_files   = filter(x -> occursin(name(wancalc), x), DFC.searchdir(job, "upre"))
            upim_files   = filter(x -> occursin(name(wancalc), x), DFC.searchdir(job, "upim"))
            downre_files = filter(x -> occursin(name(wancalc), x), DFC.searchdir(job, "downre"))
            downim_files = filter(x -> occursin(name(wancalc), x), DFC.searchdir(job, "downim"))
            any(isempty.((upre_files, upim_files, downre_files, downim_files))) && continue
            points = read_points_from_xsf(upre_files[1])

            for (ur, ui, dr, di) in zip(upre_files, upim_files, downre_files, downim_files)
                push!(t_funcs, WannierFunction(ur, ui, dr, di, points))
            end
                
        else
            xsf_files = filter(x -> occursin(".xsf", x), DFC.find_files(job, name(wancalc)))
            isempty(xsf_files) && continue
           
            points = read_points_from_xsf(xsf_files[1])
            for f in xsf_files
                push!(t_funcs, WannierFunction(f, points))
            end
        end
        push!(wanfuncs, t_funcs)
    end
    return wanfuncs
end

function plot_wannierfunctions(k_filenames, chk_info, wannier_plot_supercell::NTuple{3,Int}, wan_plot_list=1:chk_info.n_wann)
    num_kpts = length(k_filenames)
    U = chk_info.U_matrix
    U_opt = permutedims(chk_info.U_matrix_opt,(2,1,3))
    tu = read_unk(k_filenames[1])
    nrx, nry, nrz = size(tu,1), size(tu,2), size(tu,3)
    supercell_xrange = -div(wannier_plot_supercell[1],2)*nrx : div(wannier_plot_supercell[1] + 1, 2)*nrx - 1
    supercell_yrange = -div(wannier_plot_supercell[2],2)*nry : div(wannier_plot_supercell[2] + 1, 2)*nry - 1
    supercell_zrange = -div(wannier_plot_supercell[3],2)*nrz : div(wannier_plot_supercell[3] + 1, 2)*nrz - 1

    nx, ny, nz = length.((supercell_xrange, supercell_yrange, supercell_zrange))

    nwfun = length(wan_plot_list)
    wfuncs_all = zeros(eltype(tu), nwfun, (wannier_plot_supercell .* (nrx, nry, nrz))...,size(tu, 5))
    n_wann = chk_info.n_wann
    r_wan      = zeros(eltype(tu), chk_info.n_wann, nrx, nry, nrz)

    p = Progress(length(chk_info.kpoints))
    @inbounds for ik = 1:num_kpts
        k = chk_info.kpoints[ik]
        unk_all = read_unk(k_filenames[ik])
        for is = 1:size(tu, 5)
            u = U[wan_plot_list, :, ik]
            u_opt = U_opt[:, :, ik]
            inc_ids = findall(!iszero, chk_info.lwindow[:, ik])
            fill!(r_wan, 0.0)
            for ib in 1:chk_info.ndimwin[ik]
                iib = inc_ids[ib]
                Threads.@threads for nz in 1:nrz
                    for ny in 1:nry
                        for nx in 1:nrx
                            for iw in 1:n_wann
                                r_wan[iw, nx, ny, nz] += u_opt[iw, ib] * unk_all[nx,ny,nz,iib,is]
                            end
                        end
                    end
                end
            end
            Threads.@threads for iisz in 1:nz
                isz = supercell_zrange[iisz]
                iz = mod1(isz, nrz)
                for iisy in 1:ny
                    isy = supercell_yrange[iisy]
                    iy = mod1(isy, nry)
                    for iisx in 1:nx
                        isx = supercell_xrange[iisx]
                        ix = mod1(isx, nrx)
                        scalfac = exp(2im*π*dot(k, Vec3((isx-1)/nrx, (isy-1)/nry, (isz-1)/nrz)))
                        for ib in 1:n_wann
                            rt = r_wan[ib, ix, iy, iz] * scalfac
                            for iw in 1:nwfun
                                wfuncs_all[iw, iisx, iisy, iisz, is] += u[iw, ib] * rt
                            end
                        end
                    end
                end
            end
        end
        next!(p)
    end
    wfuncs_all ./= num_kpts
    if size(tu, 5) == 1
        Threads.@threads for iw = 1:size(wfuncs_all, 1)
            tmaxx = 0.0
            cmod  = 1.0+0.0im
            for iisx in 1:nx
                for iisy in 1:ny
                    for iisz in 1:nz
                        w = wfuncs_all[iw, iisx, iisy, iisz, 1]
                        t = abs2(w)
                        if t > tmaxx
                            tmaxx = t
                            cmod = w
                        end
                    end
                end
            end
            cmod /= abs(cmod)
            @views wfuncs_all[:, :, :, iw, 1] ./= cmod
        end
    end
    points = [chk_info.cell * Vec3((x-1)/nrx, (y-1)/nry, (z-1)/nrz) for x in supercell_xrange, y in supercell_yrange, z in supercell_zrange]
    if size(tu,5) == 1
        wfuncs_out = Vector{WannierFunction{1, eltype(wfuncs_all).parameters[1]}}(undef, size(wfuncs_all, 1))
        @time Threads.@threads for i=1:size(wfuncs_all, 1)
            wfuncs_out[i] = WannierFunction(points, map(x -> SVector(x), view(wfuncs_all,i, :, :, :, 1)))
        end
        return wfuncs_out 
    else
        wfuncs_out = Vector{WannierFunction{2, eltype(wfuncs_all).parameters[1]}}(undef, size(wfuncs_all, 1))
        @time Threads.@threads for i=1:size(wfuncs_all, 1)
            wfuncs_out[i] = WannierFunction(points, map(x -> SVector(x), zip(view(wfuncs_all, i, :, :, :, 1), view(wfuncs_all, i, :, :, :, 2))))
        end
        return wfuncs_out 
    end
end
function generate_wannierfunctions(job::DFJob, supercell::NTuple{3,Int}, args...)
    unk_files = DFC.searchdir(job, "UNK")
    wan_calc  = getfirst(x -> DFC.package(x)==Wannier90, DFC.inputs(job))
    chk_info  = read_chk(joinpath(job, "$(name(wan_calc)).chk"))
    return plot_wannierfunctions(unk_files, chk_info, supercell, args...)
end

read_unk(file) = occursin("NC", file) ? read_unk_noncollinear(file) : read_unk_collinear(file)

function read_unk_collinear(file)
    f = FortranFile(file)
    ngx, ngy, ngz, nk, nbnd = read(f, (Int32, 5))
    Uk = zeros(ComplexF64, ngx, ngy, ngz, nbnd, 1)
    for i in 1:nbnd
        record = FortranFiles.Record(f)
        read!(record, view(Uk,:,:,:,i,1))
        close(record)
    end
    return Uk
end

function read_unk_noncollinear(file)
    f = FortranFile(file)
    ngx, ngy, ngz, nk, nbnd = read(f, (Int32, 5))
    Uk = zeros(ComplexF64, ngx, ngy, ngz, nbnd, 2)
    for i in 1:nbnd
        record = FortranFiles.Record(f)
        read!(record, view(Uk,:,:,:,i,1))
        close(record)
        record = FortranFiles.Record(f)
        read!(record, view(Uk,:,:,:,i,2))
        close(record)
    end
    return Uk
end
