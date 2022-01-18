import DFControl.Utils: searchdir, strip_split, getfirst
using DFControl.Structures: ismagnetic
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
write_xsf(filename::String, wfc::Wfc3D{T}) where T<:AbstractFloat


Writes the real part of the Wfc3D to a .xsf file that is readable by XCrysden or VESTA.
"""
function write_xsf(filename::String, wfc, structure; value_func=x -> norm(x))
    open(filename,"w") do f
        origin = wfc.points[1,1,1]
        write(f,"# Generated from PhD calculations\n")
        write(f, "CRYSTAL\n")
        c = ustrip.(structure.cell')
        write(f, "PRIMVEC\n")
        write(f, "$(c[1,1]) $(c[1,2]) $(c[1,3])\n")
        write(f, "$(c[2,1]) $(c[2,2]) $(c[2,3])\n")
        write(f, "$(c[3,1]) $(c[3,2]) $(c[3,3])\n")
        write(f, "CONVVEC\n")
        write(f, "$(c[1,1]) $(c[1,2]) $(c[1,3])\n")
        write(f, "$(c[2,1]) $(c[2,2]) $(c[2,3])\n")
        write(f, "$(c[3,1]) $(c[3,2]) $(c[3,3])\n")
        write(f, "PRIMCOORD\n")
        write(f, "$(length(structure.atoms)) 1\n")
        for at in structure.atoms
            n = at.element.symbol
            p = ustrip.(at.position_cart)
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

@doc raw"""
	readhami(chk, eig_file::AbstractString)

Reads `eig_file` and uses the w90 checkpoint info in `chk` to return a vector of TbBlocks with the hopping parameters of the Wannier Tight Binding Hamiltonian.
"""
function readhami(chk, eig_file::AbstractString)
    v_mat = chk.V_matrix
    eigvals = read_eig(eig_file)
    R_cryst, degens = chk.ws_R_cryst, chk.ws_degens
    ws_shifts, ws_nshifts = chk.ws_shifts_cryst, chk.ws_nshifts
    Hq = map(1:length(chk.kpoints)) do ik
        v = v_mat[1:num_states(chk, ik), 1:chk.n_wann, ik]

        return v' * diagm(eigvals[disentanglement_range(chk, ik), ik]) * v
    end
    c = chk.cell'
    LT = eltype(c)
    T = eltype(eigvals)
    Hr_t = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for R in R_cryst]
    fourier_q_to_R(chk.kpoints, R_cryst) do iR, ik, phase
        @inbounds Hr_t[iR] .+= phase .* Hq[ik]
    end
    for o in Hr_t
        o ./= length(chk.kpoints)
    end

    out = [TbBlock(R, c*R, zeros(ComplexF64, chk.n_wann, chk.n_wann), zeros(ComplexF64, chk.n_wann, chk.n_wann)) for R in R_cryst]

    for (iR, (h, R, shifts, nshifts, d)) in enumerate(zip(Hr_t, R_cryst, ws_shifts, ws_nshifts, degens))
        for i in eachindex(h)
            ns = nshifts[i]
            frac = 1/(ns * d)
            for is = 1:ns
                rcryst = R + shifts[i][is]
                h1 = out[rcryst]
                if h1 === nothing
                    h1 = TbBlock{T, LT, Matrix{Complex{T}}}(rcryst, c*rcryst, zeros(ComplexF64, chk.n_wann, chk.n_wann), zeros(ComplexF64, chk.n_wann, chk.n_wann))
                    push!(out, h1)
                end
                h1.block[i] += h[i] * frac
                h1.tb_block[i] = h[i]
            end
        end
    end
    return out
end

#super not optimized
#TODO Test: new wigner seitz shift stuff
@doc raw"""
	read_colin_hami(up_chk, down_chk, up_eig_file::AbstractString, down_eig_file::AbstractString)

Returns the colinear TbHami representing the up-down blocks of the Wannier Tight Binding Hamiltonian.
"""
function read_colin_hami(up_chk, down_chk, up_eig_file::AbstractString, down_eig_file::AbstractString)
	uphami   = readhami(up_chk, up_eig_file)
	downhami = readhami(down_chk, down_eig_file)
	dim = blocksize(uphami)
	@assert dim == blocksize(downhami) "Specified files contain Hamiltonians with different dimensions of the Wannier basis."

	u1 = uphami[1]
	d1 = downhami[u1.R_cryst]
	
	first = TbBlock(u1.R_cryst,
	                u1.R_cart,
	                ColinMatrix(block(u1), block(d1)),
	                ColinMatrix(u1.tb_block, d1.tb_block))

	outhami  = [first]
	for u in uphami[2:end]
    	d = downhami[u.R_cryst]
    	if d !== nothing
    		push!(outhami, TbBlock(u.R_cryst, u.R_cart, ColinMatrix(block(u), block(d)), ColinMatrix(u.tb_block, d.tb_block)))
		end
	end
	return outhami
end

"""
    readhami(job::Job)

Goes through the job and will attempt to read the hamiltonian files.
If it finds a colinear calculation in the job it will read the up and down hamiltonians,
if the job was either nonmagnetic or noncolinear it will read only one hamiltonian file (there should be only one).
"""
function readhami(job::Job)

    @assert any(x -> x isa Calculation{Wannier90}, job.calculations) "No wannier90 calculations found in the job."

    seedname = getfirst(x -> x isa Calculation{Wannier90}, job.calculations).name
    jld_file = joinpath(job, "hami.jld2")
    if ispath(jld_file)
        return DFC.load(jld_file)["hami"]
    end
	eig_files = reverse(searchdir(job, ".eig"))
	chk_files = reverse(searchdir(job, ".chk"))
	@assert !isempty(eig_files) "No eig files ($(seedname).eig) found."
	@assert !isempty(chk_files) "No chk files ($(seedname).chk) found."
	if !DFC.Jobs.runslocal(job)
        tdir = mkpath(tempname())
    	for f in [eig_files; chk_files]
        	fname = splitpath(f)[end]
        	DFC.Servers.pull(job, fname, joinpath(tdir, fname))
    	end
    	eig_files = reverse(searchdir(tdir, ".eig"))
    	chk_files = reverse(searchdir(tdir, ".chk"))
	end
	if DFC.Structures.ismagnetic(job.structure)
    	if !DFC.Structures.iscolin(job.structure) || any(x -> DFC.Calculations.hasflag(x, :lspinorb) && x[:lspinorb], job.calculations)
    		return make_noncolin.(readhami(read_chk(chk_files[1]), joinpath(job, eig_files[1])))
    	else
    		return read_colin_hami(read_chk.(chk_files)..., eig_files...)
    	end
	else
		return readhami(read_chk(chk_files[1]), joinpath(job, eig_files[1]))
	end
end

"""
read_rmn_file(filename::String, structure::AbstractStructure)

Returns and array of tuples that define the dipoles between the Wannier functions in different unit cells.
"""
function read_rmn_file(filename::String, structure::DFC.Structures.Structure)
    open(filename) do  f
        out = RmnBlock{Float64}[]
        readline(f)
        n_wanfun = parse(Int, readline(f))
        readline(f)
        while !eof(f)
            l= split(readline(f))
            R_cryst = Vec3(parse.(Int, l[1:3]))
            block = getfirst(x -> x.R_cryst == R_cryst, out)

            if block == nothing
                block = RmnBlock{Float64}(cell(structure)' * R_cryst, R_cryst, Matrix{Point3{Float64}}(I, n_wanfun, n_wanfun))
                push!(out, block)
            end
            dipole = Point3(parse.(Float64, l[6:2:10]))
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
# end

# function write_exchanges(filename::String, structure::Structure)
#     open(filename, "w") do f
#         exchanges = data(structure)[:exchanges]
#         for (i, atom1) in enumerate(atoms(structure))
#             for (i2, atom2) in enumerate(atoms(structure)[i + 1:end])
#                 J = exchange_between(atom1, atom2, exchanges)
#                 if J != 0
#                     write(f, "$i:$(name(atom1)) -> $(i2+i):$(name(atom2)) $J\n")
#                 end
#             end
#         end
#     end
# end

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
function read_chk(filename)
    f = FortranFile(filename)
    header = String(read(f, FString{33}))
    n_bands = Int(read(f, Int32))
    n_excluded_bands = Int(read(f, Int32))
    if n_excluded_bands != 0
        read(f, (Int32, n_excluded_bands))
    else
        read(f)
        read(f)
    end
    real_lattice = 1DFControl.angstrom.*(Mat3(read(f, (Float64, 3, 3))...))
    recip_lattice = K_CART_TYPE{Float64}.(Mat3(read(f, (Float64, 3, 3))...)')
    n_kpoints = read(f, Int32)
    mp_grid = Vec3(Int.(read(f, (Int32, 3)))...)
    kpt_lattice_t = read(f, (Float64, 3, n_kpoints))
    kpt_lattice = [Vec3(kpt_lattice_t[:, i]...) for i = 1:size(kpt_lattice_t)[2]]
    k_nearest_neighbors = read(f, Int32)
 
    n_wann = read(f, Int32)
    chkpt = strip(String(read(f, FString{20})))
    have_disentangled = read(f, Int32) != 0 ? true : false
    if have_disentangled
        omega_invariant = read(f, Float64)
        lwindow = map(x-> x!=0 ? true : false, read(f, (Int32, n_bands, n_kpoints)))
        ndimwin = read(f, (Int32, n_kpoints))
        U_matrix_opt = read(f, (Complex{Float64}, n_bands, n_wann, n_kpoints))
    else
        omega_invariant = 0.0
        lwindow = fill(true, n_bands, n_kpoints)
        ndimwin = fill(n_wann, n_kpoints)
        U_matrix_opt= [i == j ? 1.0im : 0.0im for i=1:n_bands, j=1:n_wann, ik = 1:n_kpoints]
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
    wb = nothing
    try
        wb = read(f, (Float64, k_nearest_neighbors)) #this requires patched w90
    catch
        @warn "neighbor weights not found, berry calculations won't work. Patch your w90 if this functionality is wanted"
        wb = nothing
    end
    ws_R_cryst, ws_degens = wigner_seitz_points(mp_grid, metrics(ustrip.(real_lattice), ustrip.(recip_lattice)).real)
    ws_shifts_cryst, ws_nshifts = find_wigner_seitz_shifts(ws_R_cryst, wannier_centers, real_lattice, mp_grid)
    return (
        n_bands=n_bands,
        n_excluded_bands=n_excluded_bands,
        cell=real_lattice',
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
        wannier_spreads=wannier_spreads,
        ws_R_cryst = ws_R_cryst,
        ws_degens = ws_degens,
        ws_shifts_cryst = ws_shifts_cryst,
        ws_nshifts = ws_nshifts
    )
end
function read_chk(job::Job)
    if DFC.iscolin(job.structure)
        return map(s -> read_chk(joinpath(job, "$(name(getfirst(x -> eltype(x)==Wannier90&& x[:spin] == s, job.calculations))).chk")), ["up", "down"])
    else
        return read_chk(joinpath(job, "$(name(getfirst(x -> eltype(x)==Wannier90, job.calculations))).chk"))
    end
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

metrics(chk) = metrics(ustrip.(chk.cell), ustrip.(chk.recip_cell))
function metrics(cell, recip_cell)
    real  = zeros(3, 3)
    recip = zeros(3, 3)
    for j in 1:3, i in 1:j
        for l in 1:3
            real[i, j]  += cell[i, l]  * cell[j, l]
            recip[i, j] += recip_cell[i, l] * recip_cell[j, l]
        end
        if i < j
            real[j, i]  = real[i, j]
            recip[j, i] = recip[j, i]
        end
    end
    return (real = real, recip = recip)
end

#This is a straight translation from the function in W90, this give the wigner_seitz R points
# The point of this is to determine the R_cryst but also the degeneracies i.e. the periodic images that have
# the exact same distance and will thus have exactly the same Tb hamiltonian block.
# This means that if one would be interpolating kpoings without dividing by the degeneracies, the periodic images
# would be "Double counted", which is why we divide by degen. In the actual tb hamiltonian this is fine though, no division needed.
wigner_seitz_points(chk) = wigner_seitz_points(chk.mp_grid, metrics(chk).real)
function wigner_seitz_points(mp_grid, real_metric)
    nrpts   = 0
    r_degens  = Int[]
    r = Vec3{Int}[]
    for n1 in -mp_grid[1] : mp_grid[1], n2 in -mp_grid[2] : mp_grid[2], n3 in -mp_grid[3] : mp_grid[3]
        R = Vec3(n1, n2, n3)
        dist_R0  = 0.0
        min_dist = typemax(Float64)
        ndegen   = 1
        best_R = copy(R)
        for i1 in -2:2, i2 in -2:2, i3 in -2:2
            ndiff = R .- Vec3(i1, i2, i3) .* mp_grid
            dist = ndiff' * real_metric * ndiff
            if abs(dist - min_dist) < 1e-7
                ndegen += 1
            elseif dist < min_dist
                min_dist = dist
                ndegen   = 1
            end
            if i1 == i2 == i3 == 0
                dist_R0 = dist
            end
        end
        # Only if R is actually the smallest distance it gets added to the R_cryst.
        if abs(min_dist - dist_R0) < 1e-7
            push!(r, R)
            push!(r_degens, ndegen)
        end
    end
    return r, r_degens
end

const WS_DISTANCE_TOL = 1e-5
find_wigner_seitz_shifts(chk) = find_wigner_seitz_shifts(chk.ws_R_cryst, chk.wannier_centers, chk.cell, chk.mp_grid)
function find_wigner_seitz_shifts(R_cryst, wannier_centers, cell, mp_grid)
    nwann = length(wannier_centers)
    ws_shifts_cryst = [[Vec3{Int}[zero(Vec3{Int})] for i=1:nwann, j=1:nwann] for iR = 1:length(R_cryst)]
    ws_nshifts      = [zeros(Int, nwann,nwann) for iR = 1:length(R_cryst)]
    c = ustrip.(cell')
    ic = inv(c)
    for (iR, R) in enumerate(R_cryst)
        r_cart = c * R
        for i in 1:nwann, j=1:nwann
            best_r_cart = -wannier_centers[i] + r_cart + wannier_centers[j]
            nr = norm(best_r_cart)

            r_cryst = ic * best_r_cart

            for l in -3:3, m in -3:3, n in -3:3
                lmn = Vec3(l, m, n)
                test_r_cryst = r_cryst + lmn .* mp_grid
                test_r_cart  = c * test_r_cryst
                if norm(test_r_cart) < nr
                    best_r_cart = test_r_cart
                    nr = norm(test_r_cart)
                    ws_shifts_cryst[iR][i, j][1] = lmn .* mp_grid
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
                    test_r_cryst = best_r_cryst + lmn .* mp_grid
                    test_r_cart  = c * test_r_cryst
                    if abs(norm(test_r_cart) - nr) < WS_DISTANCE_TOL
                        ws_nshifts[iR][i, j] += 1
                        if ws_nshifts[iR][i, j] == 1
                            ws_shifts_cryst[iR][i, j][ws_nshifts[iR][i, j]] = orig_shift + lmn .* mp_grid
                        else
                            push!(ws_shifts_cryst[iR][i, j], orig_shift + lmn .* mp_grid)
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

function read_KBonds(nnkpfile::AbstractString)
    open(nnkpfile, "r") do f
        line = readline(f)
        while line != "begin recip_lattice"
            line = readline(f)
        end
        recip_cell = Matrix{K_CART_TYPE{Float64}}(undef, 3, 3)
        recip_cell[:, 1] = parse.(Float64, split(readline(f))) .* 1/1DFControl.angstrom
        recip_cell[:, 2] = parse.(Float64, split(readline(f))) .* 1/1DFControl.angstrom
        recip_cell[:, 3] = parse.(Float64, split(readline(f))) .* 1/1DFControl.angstrom
        
        while line != "begin kpoints"
            line = readline(f)
        end
        nkpoints = parse(Int, strip(readline(f)))
        kpoints = Vector{Vec3{K_CART_TYPE{Float64}}}(undef, nkpoints)
        counter = 1
        line = readline(f)
        while line != "end kpoints"
            kpoints[counter] = recip_cell * Vec3(parse.(Float64, split(line)))
            line = readline(f)
            counter += 1
        end
        out = [KBond[] for i = 1:nkpoints] 
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

            vr = (recip_cell * parse(Vec3{Int}, sline[3:5]) + kpoints[ik2]) - kpoints[ik]
            push!(out[ik], KBond(ik, ik2, vr))
            counter += 1
            line = readline(f)
        end
        return out
    end
end

function fill_k_neighbors!(kpoints::Vector{AbInitioKPoint{T}}, file::AbstractString, recip_cell::Mat3) where {T}
    kbonds = read_KBonds(file)
    @assert length(kbonds) == length(kpoints) "Number kpoints in seedname.nnkp doesn't match with the number of kpoints in seedname.chk."
    for ik = 1:length(kbonds)
        nntot = length(kbonds[ik])
        kpoints[ik].overlaps = [Matrix{Complex{T}}(undef, 0, 0) for ib = 1:nntot]
        kpoints[ik].hamis = [Matrix{Complex{T}}(undef, 0, 0) for ib = 1:nntot]
        kpoints[ik].neighbors = kbonds[ik]
        kpoints[ik].uHu = Matrix{Matrix{Complex{T}}}(undef, nntot, nntot)
    end
    return kpoints
end

function read_wannier_functions(job)
    wancalc_ids = findall(x -> eltype(x) == Wannier90, job.calculations)
    wanfuncs = Vector{WannierFunction}[]
    for wancalc in job.calculations[wancalc_ids]
        t_funcs = WannierFunction[]
        # here we assume that if it's a spinor calculation, and all the upre, upim, downre, downim are there
        if DFC.hasflag(wancalc, :spinors) && wancalc[:spinors]
            upre_files   = filter(x -> occursin(name(wancalc), x), searchdir(job, "upre"))
            upim_files   = filter(x -> occursin(name(wancalc), x), searchdir(job, "upim"))
            downre_files = filter(x -> occursin(name(wancalc), x), searchdir(job, "downre"))
            downim_files = filter(x -> occursin(name(wancalc), x), searchdir(job, "downim"))
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

function read_spn(filename)
    f = FortranFile(filename)
    read(f)
    nbnd, nk = read(f, (Int32, 2))
    Sx, Sy, Sz = [zeros(ComplexF64, nbnd, nbnd) for k = 1:nk],
                 [zeros(ComplexF64, nbnd, nbnd) for k = 1:nk],
                 [zeros(ComplexF64, nbnd, nbnd) for k = 1:nk]
    for ik in 1:nk
        t = read(f, (ComplexF64, 3, div(nbnd * (nbnd + 1), 2)))
        counter = 1
        for ib1 in 1:nbnd, ib2 in 1:ib1
            Sx[ik][ib2, ib1] = t[1, counter]
            Sx[ik][ib1, ib2] = conj(t[1, counter])
            Sy[ik][ib2, ib1] = t[2, counter] 
            Sy[ik][ib1, ib2] = conj(t[2, counter])
            Sz[ik][ib2, ib1] = t[3, counter] 
            Sz[ik][ib1, ib2] = conj(t[3, counter])
            counter += 1
        end
    end
    return (Sx, Sy, Sz)
end

function S_R(chk, Sx, Sy, Sz)
    #q is in wannier gauge
    nk = length(chk.kpoints)
    Sx_q = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i=1:nk]
    Sy_q = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i=1:nk]
    Sz_q = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i=1:nk]
    vmat = chk.V_matrix
    nwann = chk.n_wann
    for ik in 1:nk
        v = vmat[1:num_states(chk, ik), 1:nwann, ik]
        disr = disentanglement_range(chk, ik)
        Sx_q[ik] = v' * Sx[ik][disr, disr] * v
        Sy_q[ik] = v' * Sy[ik][disr, disr] * v
        Sz_q[ik] = v' * Sz[ik][disr, disr] * v
    end
    R_cryst, degens = chk.ws_R_cryst, chk.ws_degens
    nR = length(R_cryst) 
    Sx_R = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i=1:nR]
    Sy_R = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i=1:nR]
    Sz_R = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i=1:nR]

    fourier_q_to_R(chk.kpoints, R_cryst) do iR, ik, phase
        Sx_R[iR] .+= Sx_q[ik] .* phase
        Sy_R[iR] .+= Sy_q[ik] .* phase
        Sz_R[iR] .+= Sz_q[ik] .* phase
    end
    for iR in 1:nR
        Sx_R[iR] ./= nk
        Sy_R[iR] ./= nk
        Sz_R[iR] ./= nk
    end
    return Sx_R, Sy_R, Sz_R
end

function readspin(spn_file, chk_file)
    Sx_dft, Sy_dft, Sz_dft = read_spn(spn_file)
    return S_R(read_chk(chk_file), Sx_dft, Sy_dft, Sz_dft)
end

function readspin(job::Job)
	chk_files = reverse(searchdir(job, ".chk"))
	spn_files = reverse(searchdir(job, ".spn"))
	isempty(chk_files) && error("No .chk files found in job dir: $(job.local_dir)")
	isempty(spn_files) && error("No .spn files found in job dir: $(job.local_dir)")
	if length(chk_files) > 1
    	error("Not implemented for collinear spin-polarized calculations")
    end
    return readspin(spn_files[1], chk_files[1])
end

wan_hash(job::Job) = hash(read_chk(job))

function read_nnkp(nnkp_file) #not everything, just what I need for now
    open(nnkp_file, "r") do f
        blocks = read_wannier_blocks(f)
        nkpoints = parse(Int, blocks[:kpoints][1])
        nntot = parse(Int, blocks[:nnkpts][1])
        nnlist = Matrix{Int32}(undef, nkpoints, nntot)
        nncell = Array{Int32, 3}(undef, 3, nkpoints, nntot)
        counter = 2
        for ik = 1:nkpoints
            for nn = 1:nntot
                sl = strip_split(blocks[:nnkpts][counter])
                nnlist[ik, nn] = parse(Int, sl[2])
                for i = 1:3
                    nncell[i, ik, nn] = parse(Int, sl[i+2])
                end
                counter += 1
                    
            end
        end
        return nnlist, nncell
    end
end

function Rmn(chk, nnkp_file)
    kbonds = read_KBonds(nnkp_file)
    m_matrix = chk.m_matrix
    R_cryst, degens = chk.ws_R_cryst, chk.ws_degens
    nR = length(R_cryst)
    nwann = chk.n_wann
    nntot = chk.n_nearest_neighbors
    wb = chk.neighbor_weights
    r_R = [zeros(Vec3{ComplexF64}, nwann, nwann) for i=1:nR]
    fourier_q_to_R(chk.kpoints, R_cryst) do iR, ik, phase
        for m = 1:nwann
            for n = 1:nwann
                for nn = 1:nntot
                    r_R[iR][n, m] += m == n ? - wb[nn] .* ustrip.(kbonds[ik][nn].vr) .* imag(log(m_matrix[n, m, nn, ik])) .* phase : 1im*wb[nn] .* ustrip.(kbonds[ik][nn].vr) .* m_matrix[n, m, nn, ik] .* phase
                end
            end
        end
    end
    nk = length(chk.kpoints)
    for iR in 1:nR
        r_R[iR] ./= nk
    end
    return r_R
end

function read_wannier_blocks(f)
    out = Dict{Symbol, Any}()
    while !eof(f)
        l = readline(f)
        if occursin("begin", l)
            s = Symbol(split(l)[2])
            lines = AbstractString[]
            l = readline(f)
            while !occursin("end", l)
                push!(lines, l)
                l = readline(f)
            end
            out[s] = lines
        end
    end
    return out
end
