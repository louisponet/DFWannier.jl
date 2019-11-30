phases(kpoints::Vector{<:Vec3}, R::Vec3) = exp.(2im * π .* dot.(kpoints, (R,)))

abstract type AbstractKGrid{T} end

core_kgrid(x::AbstractKGrid) = x.core

k_cryst(x::AbstractKGrid)    = core_kgrid(x).k_cryst
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
function HamiltonianKGrid(hami::TbHami{T}, kpoints::Vector{<:Vec3}, H_function_k::Function = x -> nothing) where {T}
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
	    H_function_k(kp)
	    eigen!(kgrid.eigvals[i], kgrid.eigvecs[i], cache)
    end
    return kgrid
end

function HamiltonianKGrid(hami::TbHami{T}, nk::NTuple{3, Int}, H_function_k::Function = x -> nothing) where {T}
    k_grid  = uniform_shifted_kgrid(nk...)
    return fill_kgrid(hami, k_grid, Hfunc)
end

