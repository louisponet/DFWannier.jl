const K_CART_TYPE{T} = Quantity{T,Unitful.𝐋^-1,Unitful.FreeUnits{(Ang^-1,),Unitful.𝐋^-1,nothing}} 

phases(kpoints::Vector{<:Vec3}, R::Vec3) = exp.(2im * π .* dot.(kpoints, (R,)))

uniform_shifted_kgrid(::Type{T}, nkx::Integer, nky::Integer, nkz::Integer) where {T} =
	reshape([Vec3{T}(kx, ky, kz) for kx = 0.5/nkx:1/nkx:1, ky = 0.5/nky:1/nky:1, kz = 0.5/nkz:1/nkz:1], nkx*nky*nkz)

uniform_shifted_kgrid(nkx::Integer, nky::Integer, nkz::Integer) = uniform_shifted_kgrid(Float64, nkx, nky, nkz)

uniform_kgrid(nkx::Integer, nky::Integer, nkz::Integer) =
    reshape([Vec3{Float64}(kx, ky, kz) for kx in range(0, 1-1/nkx, length=nkx), ky in range(0, 1-1/nky, length=nky), kz in range(0, 1-1/nkz, length=nkz)], nkx*nky*nkz)

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

struct HamiltonianKGrid{T,MT<:AbstractMatrix{Complex{T}}} <: AbstractKGrid{T}
    core::CoreKGrid{T}
    # Hk::Vector{MT}
    eigvals::Vector{Vector{T}}
    eigvecs::Vector{MT}
end
HamiltonianKGrid(kpoints::Vector{<:Vec3}, args...) = HamiltonianKGrid(CoreKGrid(kpoints), args...)

@doc raw"""
	HamiltonianKGrid(hami::TbHami{T}, nk, H_function_k::Function = x -> nothing) where T
	HamiltonianKGrid(hami::TbHami{T}, k_grid, H_function_k::Function = x -> nothing) where T

Takes a k grid, calculates Hk for each of them and diagonalizes. Only the eigenvectors and eigenvalues of Hk are stored,
the H_function_k function is called on the intermediate Hk. 
"""
function HamiltonianKGrid(hami::TbHami{T}, kpoints::Vector{<:Vec3}, Hk_function::Function = x -> nothing) where {T}
	# kpoints = [KPoint(k, blocksize(hami), R, zeros_block(hami)) for k in k_grid]
	n_eigvals = max(blocksize(hami)...)
	kgrid = HamiltonianKGrid(kpoints, [zeros(T, n_eigvals) for k in kpoints],  [zeros_block(hami) for k in kpoints])
	nk    = length(kpoints)
	calc_caches = [EigCache(block(hami[1])) for i=1:nthreads()]
    @threads for i=1:nk
	    tid = threadid()
	    kp = kgrid.eigvecs[i]
	    cache = calc_caches[tid]
	    #= kp.eigvecs is used as a temporary cache to store H(k) in. Since we
	    don't need H(k) but only Hvecs etc, this is ok.
	    =#
	    Hk!(kp, hami, k_cryst(kgrid)[i])
	    Hk_function(kp)
	    eigen!(kgrid.eigvals[i], kgrid.eigvecs[i], cache)
    end
    return kgrid
end

function HamiltonianKGrid(hami::TbHami{T}, nk::NTuple{3, Int}, H_function_k::Function = x -> nothing) where {T}
    k_grid  = uniform_shifted_kgrid(nk...)
    return fill_kgrid(hami, k_grid, Hfunc)
end

abstract type KPoint{T<:AbstractFloat} end

struct KBond{T<:AbstractFloat} # All matrices/operators are in wannier gauge, i.e. block-like gauge
    k_id1   ::Int #TODO: Optimize, can we drop these?
    k_id2   ::Int
    vr      ::Vec3{K_CART_TYPE{T}}
end

@with_kw mutable struct AbInitioKPoint{T} <: KPoint{T}
    k_cryst  ::Vec3{T}
    k_cart   ::Vec3{K_CART_TYPE{T}}
    eigvals  ::Vector{T}        = T[] #original eigenvalues, in hamiltonian gauge
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

function AbInitioKGrid(::Type{T},
                       eig_filename::AbstractString,
                       chk_filename::AbstractString,
                       nnkp_filename::AbstractString,
                       mmn_filename::AbstractString,
                       uHu_filename::AbstractString) where {T}
    eigenvalues = read_eig(eig_filename) 
    wannier_chk_params = read_chk(chk_filename)
    kpoints = [AbInitioKPoint{T}(k_cryst = k,
                                  k_cart  = wannier_chk_params.recip_cell * k,
                                  eigvals = eigenvalues[:, i]) for (i, k) in enumerate(wannier_chk_params.kpoints)]
    fill_k_neighbors!(kpoints, nnkp_filename, wannier_chk_params.recip_cell)
    fill_overlaps!(kpoints, mmn_filename, uHu_filename, wannier_chk_params)
    
    return AbInitioKGrid(StructArray(kpoints), wannier_chk_params.neighbor_weights)
end

AbInitioKGrid(eig_filename::AbstractString, chk_filename::AbstractString, nnkp_filename::AbstractString, mmn_filename::AbstractString, uHu_filename::AbstractString) =
    AbInitioKGrid(Float64, eig_filename, chk_filename, nnkp_filename, mmn_filename, uHu_filename)

function AbInitioKGrid(job::DFJob)
    wancalc = getfirst(x->x isa DFInput{Wannier90}, DFControl.inputs(job))
    if wancalc === nothing
        error("Coulnd't find a wannier calculation in job $job.")
    end
    wname = name(wancalc)

    wan_file = ext -> begin
        files = DFControl.find_files(job, "$wname.$ext")
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

