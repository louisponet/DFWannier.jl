const K_CART_TYPE{T} = Quantity{T,Unitful.𝐋^-1,Unitful.FreeUnits{(Ang^-1,),Unitful.𝐋^-1,nothing}} 

phases(kpoints::Vector{<:Vec3}, R::Vec3) = exp.(2im * π .* dot.(kpoints, (R,)))

uniform_shifted_kgrid(::Type{T}, nkx::Integer, nky::Integer, nkz::Integer) where {T} =
	reshape([Vec3{T}(kx, ky, kz) for kx = 0.5/nkx:1/nkx:1, ky = 0.5/nky:1/nky:1, kz = 0.5/nkz:1/nkz:1], nkx*nky*nkz)

uniform_shifted_kgrid(nkx::Integer, nky::Integer, nkz::Integer) = uniform_shifted_kgrid(Float64, nkx, nky, nkz)

uniform_kgrid(nkx::Integer, nky::Integer, nkz::Integer) =
    reshape([Vec3{Float64}(kx, ky, kz) for kz in range(0, 1-1/nkz, length=nkz), ky in range(0, 1-1/nky, length=nky), kx in range(0, 1-1/nkx, length=nkx)], nkx*nky*nkz)

abstract type AbstractKGrid{T} end

core_kgrid(x::AbstractKGrid) = x.core

k_cryst(x::AbstractKGrid)    = core_kgrid(x).k_cryst
k_cryst(x::Vec3) = x
# k_cart(x::AbstractKGrid)     = core_kgrid(x).k_cart
# phase(x::AbstractKGrid)      = core_kgrid(x).phase
Base.length(kgrid::AbstractKGrid) = length(core_kgrid(kgrid))


struct CoreKGrid{T} <: AbstractKGrid{T}
    k_cryst::Vector{Vec3{T}}
    # k_cart ::Vector{Vec3{T}} #not needed for now
    # phase  ::Vector{Complex{T}}
end

core_kgrid(x::CoreKGrid) = x

Base.length(x::CoreKGrid) = length(x.k_cryst)

abstract type KPoint{T<:AbstractFloat} end

struct KBond{T<:AbstractFloat} # All matrices/operators are in wannier gauge, i.e. block-like gauge
    k_id1   ::Int #TODO: Optimize, can we drop these?
    k_id2   ::Int
    vr      ::Vec3{K_CART_TYPE{T}}
end

@with_kw mutable struct AbInitioKPoint{T} <: KPoint{T}
    k_cryst  ::Vec3{T}
    k_cart   ::Vec3{K_CART_TYPE{T}}
    eigvals  ::Vector{T}         #original eigenvalues, in hamiltonian gauge
    H        ::Matrix{Complex{T}} # Hk in wannier gauge 
    neighbors::Vector{KBond{T}} = KBond{T}[]
    overlaps ::Vector{Matrix{Complex{T}}} = Matrix{Complex{T}}[] #already in wannier gauge
    hamis    ::Vector{Matrix{Complex{T}}} = Matrix{Complex{T}}[] #Hamiltonian element between the block-like states in wannier gauge
    uHu      ::Matrix{Matrix{Complex{T}}} = Matrix{Matrix{Complex{T}}}(undef, 0, 0)
end

k_cryst(k::AbInitioKPoint) = k.k_cryst

struct AbInitioKGrid{T, SA} <: AbstractKGrid{T}
    kpoints::SA
    neighbor_weights::Vector{T} #ordered in the same way as neighbors in kpoints
end

num_states(chk, ik) = chk.ndimwin[ik]

function disentanglement_range(chk, ik)
    first_band_id = findfirst(view(chk.lwindow, :, ik))
    return first_band_id : first_band_id + num_states(chk, ik) - 1
end

function AbInitioKGrid(::Type{T},
                       eig_filename::AbstractString,
                       chk_filename::AbstractString,
                       nnkp_filename::AbstractString,
                       mmn_filename::AbstractString,
                       uHu_filename::AbstractString) where {T}
    eigenvalues = read_eig(eig_filename) 
    wannier_chk_params = read_chk(chk_filename)
    v_mat = wannier_chk_params.V_matrix
    nwann = wannier_chk_params.n_wann
    kpoints = map(enumerate(wannier_chk_params.kpoints)) do (ik, k)
        @views v = v_mat[1:num_states(wannier_chk_params, ik), 1:nwann, ik]
        eigvals = eigenvalues[:, ik]
        return AbInitioKPoint{T}(k_cryst = k,
                                 k_cart  = wannier_chk_params.recip_cell * k,
                                 eigvals = eigenvalues[:, ik],
                                 H       = v' * diagm(eigvals[disentanglement_range(wannier_chk_params, ik)]) * v) 

    end
    # kpoints = [AbInitioKPoint{T}(k_cryst = k,
    #                               k_cart  = wannier_chk_params.recip_cell * k,
    #                               eigvals = eigenvalues[:, i]) for (i, k) in enumerate(wannier_chk_params.kpoints)]
    fill_k_neighbors!(kpoints, nnkp_filename, wannier_chk_params.recip_cell)
    fill_overlaps!(kpoints, mmn_filename, uHu_filename, wannier_chk_params)
    
    return AbInitioKGrid(StructArray(kpoints), wannier_chk_params.neighbor_weights)
end

AbInitioKGrid(eig_filename::AbstractString, chk_filename::AbstractString, nnkp_filename::AbstractString, mmn_filename::AbstractString, uHu_filename::AbstractString) =
    AbInitioKGrid(Float64, eig_filename, chk_filename, nnkp_filename, mmn_filename, uHu_filename)

function AbInitioKGrid(job::DFJob)
    wancalc = getfirst(x->x isa DFInput{Wannier90}, DFC.inputs(job))
    if wancalc === nothing
        error("Coulnd't find a wannier calculation in job $job.")
    end
    wname = name(wancalc)

    wan_file = ext -> begin
        files = DFC.searchdir(job, "$wname.$ext")
        if isempty(files)
            error("Couldn't find $wname.$ext in job directory $(job.local_dir).")
        end
        return files[1]
    end

    return AbInitioKGrid(wan_file("eig"), wan_file("chk"), wan_file("nnkp"), wan_file("mmn"), wan_file("uHu"))
end

n_wannier_functions(grid::AbInitioKGrid) = size(grid.kpoints.overlaps[1], 1)
n_nearest_neighbors(grid::AbInitioKGrid) = length(grid.kpoints.neighbors[1])

Base.length(grid::AbInitioKGrid) = length(grid.kpoints)

function metrics(chk)
    real  = zeros(3, 3)
    recip = zeros(3, 3)
    cell = ustrip.(chk.cell)
    recip_cell = ustrip.(chk.recip_cell)
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
function wigner_seitz_points(chk)
    mp_grid = chk.mp_grid
    real_metric = metrics(chk).real
    nrpts   = 0
    r_degens  = Int[]
    r = Vec3{Int}[]
    for n1 in -mp_grid[1] : mp_grid[1], n2 in -mp_grid[2] : mp_grid[2], n3 in -mp_grid[3] : mp_grid[3]
        R = Vec3(n1, n2, n3)
        dist_R0  = 0.0
        min_dist = typemax(Float64)
        ndegen   = 0
        for i1 in -2:2, i2 in -2:2, i3 in -2:2
            ndiff = R .- Vec3(i1, i2, i3) .* mp_grid
            dist = ndiff' * real_metric * ndiff
            if dist < min_dist
                if abs(dist - min_dist) < 1e-7
                    ndegen += 1
                else
                    min_dist = dist
                    ndegen   = 0
                end
            elseif abs(dist - min_dist) < 1e-7
                ndegen += 1
            end
            if i1 == i2 == i3 == 0
                dist_R0 = dist
            end
        end
        if abs(min_dist - dist_R0) < 1e-7
            push!(r, R)
            push!(r_degens, ndegen)
        end
    end
    return r, r_degens
end


















