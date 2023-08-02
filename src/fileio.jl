import DFControl.Utils: searchdir, strip_split, getfirst
using DFControl.Structures: ismagnetic

"""
    read_chk(chk_file)
    read_chk(job::Job)

Reads a Wannier90 .chk file returning a `NamedTuple` containing all the information.
"""
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
    real_lattice = 1DFControl.angstrom .* (Mat3(read(f, (Float64, 3, 3))...))
    recip_lattice = K_CART_TYPE{Float64}.(Mat3(read(f, (Float64, 3, 3))...)')
    n_kpoints = read(f, Int32)
    mp_grid = Vec3(Int.(read(f, (Int32, 3)))...)
    kpt_lattice_t = read(f, (Float64, 3, n_kpoints))
    kpt_lattice = [Vec3(kpt_lattice_t[:, i]...) for i in 1:size(kpt_lattice_t,2)]
    k_nearest_neighbors = read(f, Int32)

    n_wann = read(f, Int32)
    chkpt = strip(String(read(f, FString{20})))
    have_disentangled = read(f, Int32) != 0 ? true : false
    if have_disentangled
        omega_invariant = read(f, Float64)
        lwindow = map(x -> x != 0 ? true : false, read(f, (Int32, n_bands, n_kpoints)))
        ndimwin = read(f, (Int32, n_kpoints))
        U_matrix_opt = read(f, (Complex{Float64}, n_bands, n_wann, n_kpoints))
    else
        omega_invariant = 0.0
        lwindow = fill(true, n_bands, n_kpoints)
        ndimwin = fill(n_wann, n_kpoints)
        U_matrix_opt = [i == j ? 1.0im : 0.0im
                        for i in 1:n_bands, j in 1:n_wann, ik in 1:n_kpoints]
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
    wannier_centers = [Point3(wannier_centers_t[:, i]...)
                       for i in 1:size(wannier_centers_t)[2]]
    wannier_spreads = read(f, (Float64, n_wann))
    wb = nothing
    try
        wb = read(f, (Float64, k_nearest_neighbors)) #this requires patched w90
    catch
        @warn "neighbor weights not found, berry calculations won't work. Patch your w90 if this functionality is wanted"
        wb = nothing
    end
    ws_R_cryst, ws_degens = wigner_seitz_points(mp_grid,
                                                metrics(ustrip.(real_lattice),
                                                        ustrip.(recip_lattice)).real)
    ws_shifts_cryst, ws_nshifts = find_wigner_seitz_shifts(ws_R_cryst, wannier_centers,
                                                           real_lattice, mp_grid)
    return (n_bands = n_bands,
            n_excluded_bands = n_excluded_bands,
            cell = real_lattice',
            recip_cell = recip_lattice,
            n_kpoints = n_kpoints,
            mp_grid = mp_grid,
            kpoints = kpt_lattice,
            n_nearest_neighbors = k_nearest_neighbors,
            neighbor_weights = wb,
            n_wann = n_wann,
            have_disentangled = have_disentangled,
            Ω_invariant = omega_invariant,
            lwindow = lwindow,
            ndimwin = ndimwin,
            U_matrix_opt = U_matrix_opt,
            U_matrix = U_matrix,
            V_matrix = V_matrix,
            m_matrix = m_matrix,
            wannier_centers = wannier_centers,
            wannier_spreads = wannier_spreads,
            ws_R_cryst = ws_R_cryst,
            ws_degens = ws_degens,
            ws_shifts_cryst = ws_shifts_cryst,
            ws_nshifts = ws_nshifts)
end
function read_chk(job::Job)
    if DFC.iscolin(job.structure)
        return map(s -> read_chk(joinpath(job,
                                          "$(name(getfirst(x -> eltype(x)==Wannier90&& x[:spin] == s, job.calculations))).chk")),
                   ["up", "down"])
    else
        return read_chk(joinpath(job,
                                 "$(name(getfirst(x -> eltype(x)==Wannier90, job.calculations))).chk"))
    end
end

# TODO: decide whether we keep NearestNeighbors or look at this here
metrics(chk) = metrics(ustrip.(chk.cell), ustrip.(chk.recip_cell))
function metrics(cell, recip_cell)
    real  = zeros(3, 3)
    recip = zeros(3, 3)
    for j in 1:3, i in 1:j
        for l in 1:3
            real[i, j]  += cell[i, l] * cell[j, l]
            recip[i, j] += recip_cell[i, l] * recip_cell[j, l]
        end
        if i < j
            real[j, i]  = real[i, j]
            recip[j, i] = recip[j, i]
        end
    end
    return (real = real, recip = recip)
end

# This is a straight translation from the function in W90, this give the wigner_seitz R points
# The point of this is to determine the R_cryst but also the degeneracies i.e. the periodic images that have
# the exact same distance and will thus have exactly the same TB hamiltonian block.
# This means that if one would be interpolating kpoings without dividing by the degeneracies, the periodic images
# would be "Double counted", which is why we divide by degen. In the actual tb hamiltonian this is fine though, no division needed.
wigner_seitz_points(chk) = wigner_seitz_points(chk.mp_grid, metrics(chk).real)
function wigner_seitz_points(mp_grid, real_metric)
    nrpts = 0
    r_degens = Int[]
    r = Vec3{Int}[]
    for n1 in -mp_grid[1]:mp_grid[1], n2 in -mp_grid[2]:mp_grid[2],
        n3 in -mp_grid[3]:mp_grid[3]

        R        = Vec3(n1, n2, n3)
        dist_R0  = 0.0
        min_dist = typemax(Float64)
        ndegen   = 1
        best_R   = copy(R)
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
function find_wigner_seitz_shifts(chk)
    return find_wigner_seitz_shifts(chk.ws_R_cryst, chk.wannier_centers, chk.cell,
                                    chk.mp_grid)
end
function find_wigner_seitz_shifts(R_cryst, wannier_centers, cell, mp_grid)
    nwann           = length(wannier_centers)
    ws_shifts_cryst = [[Vec3{Int}[zero(Vec3{Int})] for i in 1:nwann, j in 1:nwann] for iR in 1:length(R_cryst)]
    ws_nshifts      = [zeros(Int, nwann, nwann) for iR in 1:length(R_cryst)]
    c               = ustrip.(cell')
    ic              = inv(c)
    for (iR, R) in enumerate(R_cryst)
        r_cart = c * R
        for i in 1:nwann, j in 1:nwann
            best_r_cart = -wannier_centers[i] + r_cart + wannier_centers[j]
            nr = norm(best_r_cart)

            r_cryst = ic * best_r_cart

            for l in -3:3, m in -3:3, n in -3:3
                lmn          = Vec3(l, m, n)
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
                orig_shift = ws_shifts_cryst[iR][i, j][1]
                for l in -3:3, m in -3:3, n in -3:3
                    lmn          = Vec3(l, m, n)
                    test_r_cryst = best_r_cryst + lmn .* mp_grid
                    test_r_cart  = c * test_r_cryst
                    if abs(norm(test_r_cart) - nr) < WS_DISTANCE_TOL
                        ws_nshifts[iR][i, j] += 1
                        if ws_nshifts[iR][i, j] == 1
                            ws_shifts_cryst[iR][i, j][ws_nshifts[iR][i, j]] = orig_shift +
                                                                              lmn .* mp_grid
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

"""
    read_eig(eig_file)

Reads the DFT eigenvalues from a .eig file.
"""
function read_eig(filename)
    t = readdlm(filename)
    n_bands = maximum(t[:, 1])
    n_kpoints = maximum(t[:, 2])

    #we follow order of w90 eigval matrix
    Hk = Matrix{Float64}(undef, Int(n_bands), Int(n_kpoints))
    for x in 1:size(t)[1]
        tv = t[x, :]
        Hk[Int(tv[1]), Int(tv[2])] = tv[3]
    end
    return Hk
end

"""
    read_hamiltonian(chk::NamedTuple, eigvals::Matrix)

Uses the Wannier90 chkpoint info in `chk` and DFT `eigenvals` read with [`read_eig`] to construct the [`TBHamiltonian`](@ref TBOperator).
"""
function read_hamiltonian(chk::NamedTuple, eigvals::Matrix)
    v_mat = chk.V_matrix
    R_cryst = chk.ws_R_cryst

    Hq = map(1:length(chk.kpoints)) do ik
        v = v_mat[1:num_states(chk, ik), 1:chk.n_wann, ik]
        return v' * diagm(eigvals[disentanglement_range(chk, ik), ik]) * v
    end

    c = chk.cell'

    Hr_t = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for R in R_cryst]

    fourier_q_to_R(chk.kpoints, R_cryst) do iR, ik, phase
        @inbounds Hr_t[iR] .+= phase .* Hq[ik]
    end

    for o in Hr_t
        o ./= length(chk.kpoints)
    end

    return generate_TBBlocks(chk, Hr_t)
end

function read_hamiltonian(chk_file::AbstractString, eig_file::AbstractString)
    return read_hamiltonian(read_chk(chk_file), read_eig(eig_file))
end
function read_hamiltonian(up_chk_file::AbstractString, dn_chk_file::AbstractString,
                          up_eig_file::AbstractString, dn_eig_file::AbstractString)
    return read_colin_hami(read_chk(up_chk_file), read_chk(dn_chk_file),
                           read_eig(up_eig_file), read_eig(dn_eig_file))
end
#super not optimized
#TODO Test: new wigner seitz shift stuff
"""
    read_colin_hami(up_chk, down_chk, up_eigvals::AbstractString, down_eigvals::AbstractString)

Returns the colinear TBHamiltonian representing the up-down blocks of the Wannier Tight Binding Hamiltonian.
"""
function read_colin_hami(up_chk, down_chk, up_eigvals::Matrix, down_eigvals::Matrix)
    uphami   = read_hamiltonian(up_chk, up_eigvals)
    downhami = read_hamiltonian(down_chk, down_eigvals)
    dim      = blocksize(uphami)
    @assert dim == blocksize(downhami) "Specified files contain Hamiltonians with different dimensions of the Wannier basis."

    u1 = uphami[1]
    d1 = downhami[u1.R_cryst]

    first = TBBlock(u1.R_cryst,
                    u1.R_cart,
                    ColinMatrix(block(u1), block(d1)),
                    ColinMatrix(u1.tb_block, d1.tb_block))

    outhami = [first]
    for u in uphami[2:end]
        d = downhami[u.R_cryst]
        if d !== nothing
            push!(outhami,
                  TBBlock(u.R_cryst, u.R_cart, ColinMatrix(block(u), block(d)),
                          ColinMatrix(u.tb_block, d.tb_block)))
        end
    end
    return outhami
end

"""
    read_hamiltonian(job::Job)

Goes through the job and will attempt to read the hamiltonian files.
If it finds a colinear calculation in the job it will read the up and down hamiltonians,
if the job was either nonmagnetic or noncolinear it will read only one hamiltonian file (there should be only one).
"""
function read_hamiltonian(job::Job)
    @assert any(x -> x isa Calculation{Wannier90}, job.calculations) "No wannier90 calculations found in the job."

    wcalc = getfirst(x -> x isa Calculation{Wannier90}, job.calculations)
    seedname = wcalc.name
    eig_files = filter(x -> any(y -> occursin(y.name, x), job.calculations),
                       reverse(searchdir(job, ".eig")))
    chk_files = filter(x -> any(y -> occursin(y.name, x), job.calculations),
                       reverse(searchdir(job, ".chk")))
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
    if haskey(wcalc, :spin)
        hami = read_colin_hami(read_chk.(chk_files)..., read_eig.(eig_files)...)
    elseif haskey(wcalc, :spinors)
        hami = make_noncolin.(read_hamiltonian(read_chk(chk_files[1]),
                                               read_eig(joinpath(job, eig_files[1]))))
    else
        hami = read_hamiltonian(read_chk(chk_files[1]),
                                read_eig(joinpath(job, eig_files[1])))
    end
    if !DFC.Jobs.runslocal(job)
        dir = splitdir(eig_files[1])[1]
        rm(dir; recursive = true)
    end
    return hami
end

# not used

# mutable struct ReciprocalOverlaps{T}
#     k_id            :: Int
#     neighbor_ids    :: Vector{Int}
#     b_vectors_cryst :: Vector{Vec3{Int}} #connecting vector in crystalline coordinates
#     overlaps        :: Vector{Matrix{Complex{T}}}
# end

# function ReciprocalOverlaps{T}(k_id::Int, n_nearest_neighbors::Int, nbands::Int) where {T}
#     overlap_matrices = [Matrix{Complex{T}}(undef, nbands, nbands)
#                         for i in 1:n_nearest_neighbors]
#     return ReciprocalOverlaps{T}(k_id, zeros(Int, n_nearest_neighbors),
#                                  zeros(Vec3{Int}, n_nearest_neighbors), overlap_matrices)
# end

function read_uHu(file)
    try
        uHu_file = FortranFile(file)
        read(uHu_file, FString{20})
        nbands, nkpoints, n_nearest_neighbors = read(uHu_file, (Int32, 3))
        out = [Matrix{ComplexF64}(undef, nbands, nbands)
               for i in 1:nkpoints*n_nearest_neighbors^2]
        for i in 1:nkpoints*n_nearest_neighbors^2
            out[i] = transpose(read(uHu_file, (ComplexF64, nbands, nbands)))
        end
        return out
    catch
        open(file, "r") do f
            readline(f)
            nbands, nkpoints, n_nearest_neighbors = parse.(Int, strip_split(readline(f)))
            out = [Matrix{ComplexF64}(undef, nbands, nbands)
                   for i in 1:nkpoints*n_nearest_neighbors^2]
            for i in 1:nkpoints*n_nearest_neighbors^2
                for j in 1:nbands
                    for k in 1:nbands
                        out[i][j, k] = complex(parse.(Float64, strip_split(readline(f)))...)
                    end
                end
            end
            return out
        end
    end
end

function fill_overlaps!(grid::Vector{AbInitioKPoint{T}}, mmn_filename::AbstractString,
                        uHu_filename::AbstractString, wannier_chk_params) where {T}
    num_wann = wannier_chk_params.n_wann
    uHu_file = FortranFile(uHu_filename)
    read(uHu_file, FString{20})
    read(uHu_file, (Int32, 3))

    open(mmn_filename, "r") do f
        readline(f) #header 
        nbands, nkpoints, n_nearest_neighbors = parse.(Int, strip_split(readline(f)))
        #pre setup uHu
        for k in grid
            k.uHu = [Matrix{Complex{T}}(undef, num_wann, num_wann)
                     for m in 1:n_nearest_neighbors, n in 1:n_nearest_neighbors]
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
            S12 = overlap_ab_initio_gauge[disentanglement_range_k1,
                                          disentanglement_range_k2]

            kpoint = grid[ik] 

            vr = (wannier_chk_params.recip_cell * parse(Vec3{Int}, sline[3:5]) +
                  grid[ik2].k_cart) - kpoint.k_cart

            V1_T = V1'
            S12_V2 = S12 * V2
            kpoint.overlaps[cur_neighbor] = V1_T * S12_V2
            k_eigvals_mat = diagm(kpoint.eigvals[disentanglement_range_k1])
            kpoint.hamis[cur_neighbor] = V1_T * k_eigvals_mat * S12_V2
            neighbor_counter += 1
            for nearest_neighbor2 in 1:n_nearest_neighbors
                ik3               = kpoint.neighbors[nearest_neighbor2].k_id2
                first_band_id_ik3 = findfirst(wannier_chk_params.lwindow[:, ik3])
                num_states_ik3    = wannier_chk_params.ndimwin[ik3]

                V3 = wannier_chk_params.V_matrix[1:num_states_ik3, 1:num_wann, ik3]

                uHu_k2_k3 = transpose(read(uHu_file, (ComplexF64, nbands, nbands)))
                disentanglement_range_k3 = first_band_id_ik3:first_band_id_ik3+num_states_ik3-1
                kpoint.uHu[nearest_neighbor2, cur_neighbor] = V3' *
                                                              uHu_k2_k3[disentanglement_range_k3,
                                                                        disentanglement_range_k2] *
                                                              V2
            end
        end
        return grid
    end
end
 
function fill_k_neighbors!(kpoints::Vector{AbInitioKPoint{T}}, file::AbstractString,
                           recip_cell::Mat3) where {T}
    kbonds = read_nnkp(file).kbonds
    @assert length(kbonds) == length(kpoints) "Number kpoints in seedname.nnkp doesn't match with the number of kpoints in seedname.chk."
    for ik in 1:length(kbonds)
        nntot = length(kbonds[ik])
        kpoints[ik].overlaps = [Matrix{Complex{T}}(undef, 0, 0) for ib in 1:nntot]
        kpoints[ik].hamis = [Matrix{Complex{T}}(undef, 0, 0) for ib in 1:nntot]
        kpoints[ik].neighbors = kbonds[ik]
        kpoints[ik].uHu = Matrix{Matrix{Complex{T}}}(undef, nntot, nntot)
    end
    return kpoints
end

function read_unk(file)
    return occursin("NC", file) ? read_unk_noncollinear(file) : read_unk_collinear(file)
end

function read_unk_collinear(file)
    f = FortranFile(file)
    ngx, ngy, ngz, nk, nbnd = read(f, (Int32, 5))
    Uk = zeros(ComplexF64, ngx, ngy, ngz, nbnd, 1)
    for i in 1:nbnd
        record = FortranFiles.Record(f)
        read!(record, view(Uk, :, :, :, i, 1))
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
        read!(record, view(Uk, :, :, :, i, 1))
        close(record)
        record = FortranFiles.Record(f)
        read!(record, view(Uk, :, :, :, i, 2))
        close(record)
    end
    return Uk
end

"""
    read_spn(filename)

Reads a .spn file and returns the DFT Sx, Sy, Sz. They are a `Vectors` with **nk** `Matrices` of size
(**nb**, **nb**), where **nk** is the number of _k_-points and **nb** the number of bands.
"""
function read_spn(filename)
    f = FortranFile(filename)
    read(f)
    nbnd, nk = read(f, (Int32, 2))
    Sx, Sy, Sz = [zeros(ComplexF64, nbnd, nbnd) for k in 1:nk],
                 [zeros(ComplexF64, nbnd, nbnd) for k in 1:nk],
                 [zeros(ComplexF64, nbnd, nbnd) for k in 1:nk]
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

"""
    S_R(chk, Sx, Sy, Sz)

Takes the DFT `Sx`, `Sy`, `Sz` spin matrices and constructs the [`TBSpin`](@ref TBOperator) from them.
Using the Wannier90 checkpoint information in `chk`. 
"""
function S_R(chk, Sx, Sy, Sz)
    #q is in wannier gauge
    nk = length(chk.kpoints)
    Sx_q = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i in 1:nk]
    Sy_q = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i in 1:nk]
    Sz_q = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i in 1:nk]
    vmat = chk.V_matrix
    nwann = chk.n_wann
    for ik in 1:nk
        v = vmat[1:num_states(chk, ik), 1:nwann, ik]
        disr = disentanglement_range(chk, ik)
        Sx_q[ik] = v' * Sx[ik][disr, disr] * v
        Sy_q[ik] = v' * Sy[ik][disr, disr] * v
        Sz_q[ik] = v' * Sz[ik][disr, disr] * v
    end
    R_cryst = chk.ws_R_cryst
    nR = length(R_cryst)
    Sx_R = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i in 1:nR]
    Sy_R = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i in 1:nR]
    Sz_R = [zeros(ComplexF64, chk.n_wann, chk.n_wann) for i in 1:nR]

    fourier_q_to_R(chk.kpoints, R_cryst) do iR, ik, phase
        Sx_R[iR] .+= Sx_q[ik] .* phase
        Sy_R[iR] .+= Sy_q[ik] .* phase
        return Sz_R[iR] .+= Sz_q[ik] .* phase
    end
    for iR in 1:nR
        Sx_R[iR] ./= nk
        Sy_R[iR] ./= nk
        Sz_R[iR] ./= nk
    end

    return (Sx = generate_TBBlocks(chk, Sx_R),
            Sy = generate_TBBlocks(chk, Sy_R),
            Sz = generate_TBBlocks(chk, Sz_R))
end

"""
    read_spin(chk_file, spn_file)
    read_spin(job::Job)

Reads the .spn and .chk files to generate a [`TBSpin`](@ref TBOperator) tight-binding spin operator.
"""
function read_spin(chk_file, spn_file)
    Sx_dft, Sy_dft, Sz_dft = read_spn(spn_file)
    return S_R(read_chk(chk_file), Sx_dft, Sy_dft, Sz_dft)
end

function read_spin(job::Job)
    chk_files = reverse(searchdir(job, ".chk"))
    spn_files = reverse(searchdir(job, ".spn"))
    isempty(chk_files) && error("No .chk files found in job dir: $(job.local_dir)")
    isempty(spn_files) && error("No .spn files found in job dir: $(job.local_dir)")
    if length(chk_files) > 1
        error("Not implemented for collinear spin-polarized calculations")
    end
    return readspin(chk_files[1], spn_files[1])
end

"""
    read_wannier_blocks(f)
Reads a Wannier90 file such as .nnkp and separates each begin end block into an entry in the ouput `Dict`. 
"""
function read_wannier_blocks(f)
    out = Dict{Symbol,Any}()
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

"""
    read_nnkp(nnkp_file)

Reads a Wannier90 .nnkp file and returns `(recip_cell, kpoints, kbonds)`. 
"""
function read_nnkp(nnkp_file) #not everything, just what I need for now
    open(nnkp_file, "r") do f
        blocks = read_wannier_blocks(f)
        
        recip_cell = Matrix{K_CART_TYPE{Float64}}(undef, 3, 3)
        recip_cell[:, 1] = parse.(Float64, split(blocks[:recip_lattice][1])) .* 1 / 1DFControl.angstrom
        recip_cell[:, 2] = parse.(Float64, split(blocks[:recip_lattice][2])) .* 1 / 1DFControl.angstrom
        recip_cell[:, 3] = parse.(Float64, split(blocks[:recip_lattice][3])) .* 1 / 1DFControl.angstrom
        
        nkpoints = parse(Int, blocks[:kpoints][1])
        kpoints = map(view(blocks[:kpoints], 2:nkpoints+1)) do l
            return recip_cell * Vec3(parse.(Float64, split(l)))
        end
        
        n_nearest_neighbors = parse(Int, blocks[:nnkpts][1])
        k_bonds = map(view(blocks[:nnkpts], 2:length(blocks[:nnkpts]))) do line
            sline = strip_split(line)
            ik, ik2 = parse.(Int, sline[1:2])

            vr = (recip_cell * parse(Vec3{Int}, sline[3:5]) + kpoints[ik2]) - kpoints[ik]
            return KBond(ik, ik2, Vec3(vr...))
        end
        return (recip_cell = recip_cell, kpoints = kpoints, kbonds = k_bonds)
    end
end

"""
    r_R(chk, kbonds)

Constructs the _r_ [`TBOperator`](@ref) from the Wannier90 checkpoint info `chk` and the `kbond` information that
can be read with [`read_nnkp`](@ref).
"""
function r_R(chk, kbonds)
    m_matrix = chk.m_matrix
    R_cryst = chk.ws_R_cryst
    nR = length(R_cryst)
    nwann = chk.n_wann
    nntot = chk.n_nearest_neighbors

    bshells = search_shells(chk.kpoints, Unitful.ustrip.(chk.recip_cell))
    
    wb = chk.neighbor_weights
    r_R = [zeros(Vec3{ComplexF64}, nwann, nwann) for i in 1:nR]
    @inbounds fourier_q_to_R(bshells.kpoints, R_cryst) do iR, ik, phase
        r_R_t = r_R[iR]
        for nn in 1:nntot
            w = bshells.weights[nn]
            # TODO: can be all from bshells
            kb = ustrip.(kbonds[ik][nn].vr)
            for m in 1:nwann
                for n in 1:nwann
                    mm = m_matrix[n, m, nn, ik]
                    if m == n
                        t = - w .* kb .* imag(log(mm)) .* phase
                    else
                        t = 1im * w .* kb .* mm .* phase
                    end
                    r_R_t[n, m] += t
                end
            end
        end
    end
    nk = length(chk.kpoints)
    for iR in 1:nR
        r_R[iR] ./= nk
    end
    return generate_TBBlocks(chk, r_R)
end

"""
    read_r(chk_file::AbstractString, nnkp_file::AbstractString)
    read_r(job::Job)
    
Constructs the _r_ [`TBOperator`] from the Wannier90 .chk and .nnkp files.
This requires that the `k_neighbor_weights` is written into the .chk file and might
need a patched Wannier90 version.
"""
function read_r(chk_file::AbstractString, nnkp_file::AbstractString)
     return r_R(read_chk(chk_file), read_nnkp(nnkp_file).kbonds)
end

function read_r(job::Job)
    chk_files = reverse(searchdir(job, ".chk"))
    nnkp_files = reverse(searchdir(job, ".nnkp"))
    isempty(chk_files) && error("No .chk files found in job dir: $(job.local_dir)")
    isempty(nnkp_files) && error("No .spn files found in job dir: $(job.local_dir)")
    if length(chk_files) > 1
        error("Not implemented for collinear spin-polarized calculations")
    end
    return read_r(chk_files[1], nnkp_files[1])
end

# """
#     read_rmn_file(filename::String, structure::Structure)

# Returns the [`TBRmn`](@ref) operator that defines the _r_ matrix elements between the Wannier functions in different unit cells.
# """
# function read_rmn_file(filename::String, structure::Structure)
#     open(filename) do f
#         out = RmnBlock{Float64}[]
#         readline(f)
#         n_wanfun = parse(Int, readline(f))
#         readline(f)
#         while !eof(f)
#             l = split(readline(f))
#             R_cryst = Vec3(parse.(Int, l[1:3]))
#             block = getfirst(x -> x.R_cryst == R_cryst, out)

#             if block == nothing
#                 block = RmnBlock{Float64}(cell(structure)' * R_cryst, R_cryst,
#                                           Matrix{Point3{Float64}}(I, n_wanfun, n_wanfun))
#                 push!(out, block)
#             end
#             dipole = Point3(parse.(Float64, l[6:2:10]))
#             block.block[parse.(Int, l[4:5])...] = dipole
#         end
#         return out
#     end
# end

"""
    write_xsf(filename::String, wfc::WannierFunction, str::Structure; value_func=x->norm(x))

Writes a [`WannierFunction`](@ref) and [`Structure`](https://louisponet.github.io/DFControl.jl/stable/guide/structure/) to an xsf file that is readable by XCrysden or VESTA.
The values that are written can be controlled by `value_func` that gets used on each entry of `wfc.values` and should output a single `Number`. 
"""
function write_xsf(filename::String, wfc::WannierFunction, structure::Structure;
                   value_func = x -> norm(x))
    open(filename, "w") do f
        origin = wfc.points[1, 1, 1]
        write(f, "# Generated from PhD calculations\n")
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
        write.((f,),
               ["", "BEGIN_BLOCK_DATAGRID_3D\n", "3D_FIELD\n",
                "BEGIN_DATAGRID_3D_UNKNOWN\n"])
        write(f, "$(size(wfc)[1])    $(size(wfc)[2])     $(size(wfc)[3])\n")
        write(f, "$(origin[1])   $(origin[2])   $(origin[3])\n")
        write(f,
              "$(wfc.points[end,1,1][1]-origin[1])   $(wfc.points[end,1,1][2]-origin[2])   $(wfc.points[end,1,1][3]-origin[3])\n")
        write(f,
              "$(wfc.points[1,end,1][1]-origin[1])   $(wfc.points[1,end,1][2]-origin[2])   $(wfc.points[1,end,1][3]-origin[3])\n")
        write(f,
              "$(wfc.points[1,1,end][1]-origin[1])   $(wfc.points[1,1,end][2]-origin[2])   $(wfc.points[1,1,end][3]-origin[3])\n")
        for wfp in wfc.values
            write(f, "$(value_func(wfp)) ")
        end
        write(f, "\n")
        return write.((f,), ["END_DATAGRID_3D\n", "END_BLOCK_DATAGRID_3D\n"])
    end
end

function read_w90_input(file)
    return DFControl.FileIO.wan_parse_calculation(file)
end
