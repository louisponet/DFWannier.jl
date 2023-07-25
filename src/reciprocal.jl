const K_CART_TYPE{T} = Quantity{T,Unitful.𝐋^-1,
                                Unitful.FreeUnits{(Ang^-1,),Unitful.𝐋^-1,nothing}}

phases(kpoints::Vector{<:Vec3}, R::Vec3) = exp.(-2im * π .* dot.(kpoints, (R,)))

function uniform_shifted_kgrid(::Type{T}, nkx::Integer, nky::Integer,
                               nkz::Integer, gamma_center = false) where {T}

    t = [Vec3{T}(kx, ky, kz) for kx in 0:nkx-1, ky in 0:nky-1, kz in 0:nkz-1]
    s = Vec3(nkx, nky, nkz)
    t = map(t) do x
        (x .+ 0.5) ./ s .- 0.5
    end
    if gamma_center
        shift = 0.5 .* ((s.+ 1) .% 2)./s
        t = map(t) do x
            x .+ shift
        end
    end
                               
    return reshape(t, nkx * nky * nkz)
end

function uniform_shifted_kgrid(nkx::Integer, nky::Integer, nkz::Integer, gamma_center=false)
    return uniform_shifted_kgrid(Float64, nkx, nky, nkz, gamma_center)
end

function uniform_kgrid(nkx::Integer, nky::Integer, nkz::Integer)
    return reshape([Vec3{Float64}(kx, ky, kz)
                    for kz in range(0, (1 - 1 / nkz); length = nkz),
                        ky in range(0, (1 - 1 / nky); length = nky),
                        kx in range(0, (1 - 1 / nkx); length = nkx)], nkx * nky * nkz)
end

abstract type AbstractKGrid{T} end

core_kgrid(x::AbstractKGrid) = x.core

k_cryst(x::AbstractKGrid) = core_kgrid(x).k_cryst
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
    k_id1 :: Int #TODO: Optimize, can we drop these?
    k_id2 :: Int
    vr    :: Vec3{K_CART_TYPE{T}}
end

Base.@kwdef mutable struct AbInitioKPoint{T} <: KPoint{T}
    k_cryst                              :: Vec3{T}
    k_cart                               :: Vec3{K_CART_TYPE{T}}
    eigvals                              :: Vector{T}         #original eigenvalues, in hamiltonian gauge
    H                                    :: Matrix{Complex{T}} # Hk in wannier gauge 
    neighbors::Vector{KBond{T}}          = KBond{T}[]
    overlaps::Vector{Matrix{Complex{T}}} = Matrix{Complex{T}}[] #already in wannier gauge
    hamis::Vector{Matrix{Complex{T}}}    = Matrix{Complex{T}}[] #Hamiltonian element between the block-like states in wannier gauge
    uHu::Matrix{Matrix{Complex{T}}}      = Matrix{Matrix{Complex{T}}}(undef, 0, 0)
end

k_cryst(k::AbInitioKPoint) = k.k_cryst

struct AbInitioKGrid{T,SA} <: AbstractKGrid{T}
    kpoints::SA
    neighbor_weights::Vector{T} #ordered in the same way as neighbors in kpoints
end

num_states(chk, ik) = chk.ndimwin[ik]

function disentanglement_range(chk, ik)
    first_band_id = findfirst(view(chk.lwindow, :, ik))
    return first_band_id:first_band_id+num_states(chk, ik)-1
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
        return AbInitioKPoint{T}(; k_cryst = k,
                                 k_cart = wannier_chk_params.recip_cell * k,
                                 eigvals = eigenvalues[:, ik],
                                 H = v' *
                                     diagm(eigvals[disentanglement_range(wannier_chk_params,
                                                                         ik)]) * v)
    end
    fill_k_neighbors!(kpoints, nnkp_filename, wannier_chk_params.recip_cell)
    fill_overlaps!(kpoints, mmn_filename, uHu_filename, wannier_chk_params)

    return AbInitioKGrid(StructArray(kpoints), wannier_chk_params.neighbor_weights)
end

function AbInitioKGrid(eig_filename::AbstractString, chk_filename::AbstractString,
                       nnkp_filename::AbstractString, mmn_filename::AbstractString,
                       uHu_filename::AbstractString)
    return AbInitioKGrid(Float64, eig_filename, chk_filename, nnkp_filename, mmn_filename,
                         uHu_filename)
end

function AbInitioKGrid(job::Job)
    wancalc = getfirst(x -> x isa Calculation{Wannier90}, job.calculations)
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

    return AbInitioKGrid(wan_file("eig"), wan_file("chk"), wan_file("nnkp"),
                         wan_file("mmn"), wan_file("uHu"))
end

n_wannier_functions(grid::AbInitioKGrid) = size(grid.kpoints.overlaps[1], 1)
n_nearest_neighbors(grid::AbInitioKGrid) = length(grid.kpoints.neighbors[1])

Base.length(grid::AbInitioKGrid) = length(grid.kpoints)

struct HamiltonianKGrid{T,MT<:AbstractMatrix{Complex{T}},VT<:AbstractVector{T}} <:
       AbstractKGrid{T}
       
    core::CoreKGrid{T}
    
    Hk::Vector{MT}
    eigvals::Vector{VT}
    eigvecs::Vector{MT}
end
function HamiltonianKGrid(kpoints::Vector{<:Vec3}, args...)
    return HamiltonianKGrid(CoreKGrid(kpoints), args...)
end

@doc raw"""
	HamiltonianKGrid(hami::TBHamiltonian{T}, nk, H_function_k::Function = x -> nothing) where T
	HamiltonianKGrid(hami::TBHamiltonian{T}, k_grid, H_function_k::Function = x -> nothing) where T

Takes a k grid, calculates Hk for each of them and diagonalizes. Only the eigenvectors and eigenvalues of Hk are stored,
the `H_function_k` function is called on the intermediate Hk. 
"""
function HamiltonianKGrid(hami::TBHamiltonian{T}, kpoints::Vector{<:Vec3},
                          Hk_function::Function = x -> nothing) where {T}

    # kpoints = [KPoint(k, blocksize(hami), R, zeros_block(hami)) for k in k_grid]
    n_eigvals = max(blocksize(hami)...)
    eigvals = hami[1].block isa AbstractMagneticMatrix ?
              [MagneticVector(zeros(T, n_eigvals)) for k in kpoints] :
              [zeros(T, n_eigvals) for k in kpoints]
              
    kgrid = HamiltonianKGrid(kpoints, [zeros_block(hami) for k in kpoints], eigvals,
                             [zeros_block(hami) for k in kpoints])
    nk = length(kpoints)
    calc_caches = [HermitianEigenWs(block(hami[1])) for i in 1:nthreads()]
    p = Progress(nk, 1, "Calculating H(k)...")
    
    @threads for i in 1:nk
        tid = threadid()
        
        Hk!(kgrid.eigvecs[i], hami, k_cryst(kgrid)[i])
        
        copy!(kgrid.Hk[i], copy(kgrid.eigvecs[i]))
        Hk_function(kgrid.Hk[i])
        eigen!(kgrid.eigvals[i], kgrid.eigvecs[i], calc_caches[tid])
        next!(p)
    end
    
    return kgrid
end

function Hk!(out::AbstractMatrix, tbhami::TBHamiltonian, kpoint::Vec3)
    fill!(out, zero(eltype(out)))
    
    fourier_transform(tbhami, kpoint) do i, iR, R_cart, b, fac
        @inbounds out[i] += fac * b.block[i]
    end
    
end

"""
    Hk(hamiltonian::TBHamiltonian, kpoint::Vec3)
    Hk!(hk::AbstractMatrix, hamiltonian::TBHamiltonian, kpoint::Vec3)

Constructs the reciprocal Hamiltonian at a given _k_-point.  
"""
function Hk(tbhami::TBHamiltonian, kpoint::Vec3)
    out = similar(tbhami[1].block)
    Hk!(out, tbhami, kpoint)
    return out
end

Hk(g::HamiltonianKGrid) = g.Hk
eigvecs(g::HamiltonianKGrid) = g.eigvecs
eigvals(g::HamiltonianKGrid) = g.eigvals

"Fourier transforms the tight binding hamiltonian and calls the R_function with the current index and the phase."
function fourier_transform(R_function::Function, tb_hami::TBHamiltonian{T}, kpoint::Vec3) where {T}
    
    for (iR, b) in enumerate(tb_hami)
        fac = ℯ^(2im * π * (b.R_cryst ⋅ kpoint))
        for i in eachindex(block(b))
            R_function(i, iR, b.R_cart, b, fac)
        end
    end
end

"""
    WannierBand
    
Represents a Wannier interpolated band. See also [`wannierbands`](@ref). 
""" 
mutable struct WannierBand{T<:AbstractFloat,VT<:AbstractVector} <: DFC.AbstractBand
    kpoints_cryst::Vector{Vec3{T}}
    eigvals      ::Vector{T}
    eigvec       ::Vector{VT}
end

DFControl.eigvals(b::WannierBand) = b.eigvals

function Base.show(io::IO, band::WannierBand)
    summary(io, band)
    string = """
    
    $(length(band.kpoints_cryst)) k_points: $(band.kpoints_cryst[1]) -> $(band.kpoints_cryst[end])
    mean energy: $(sum(band.eigvals)/length(band.eigvals)) eV 
    """
    println(io, string)
    return
end
    

"""
    wannierbands(hamiltonian::TBHamiltonian, kpoints::Vector{Vec3})
    wannierbands(hamiltonian::TBHamiltonian, bands::Vector{DFControl.AbstractBand}
    
Constructs the whole bandstructure for a given set of _k_-points and [`TBHamiltonian`](@ref TBOperator).
"""
function wannierbands(tbhamis::TBHamiltonian{T}, kpoints::Vector{<:Vec3}) where {T}
    matdim = blocksize(tbhamis, 2)
    kgrid  = HamiltonianKGrid(tbhamis, kpoints)
    nbnd   = size(tbhamis[1].block, 2)
    evals  = [zeros(length(kpoints)) for i in 1:nbnd]
    evecs  = [[similar(kgrid.eigvecs[1][:, 1]) for i in 1:length(kpoints)] for i in 1:nbnd]
    @threads for i in 1:length(kpoints)
        eigvals, eigvecs = kgrid.eigvals[i], kgrid.eigvecs[i]
        for e in 1:length(eigvals)
            evals[e][i] = eigvals[e]
            evecs[e][i] = eigvecs[:, e]
        end
    end
    return [WannierBand{T,eltype(evecs[1])}(kpoints, evals[i],
                                            evecs[i]) for i in 1:nbnd]
end
function wannierbands(tbhamis, dfbands::Vector{<:DFC.AbstractBand})
    return wannierbands(tbhamis, dfbands[1].k_points_cryst)
end
function wannierbands(tbhamis, dfbands::Union{NamedTuple,Tuple})
    return wannierbands(tbhamis, dfbands[1][1].k_points_cryst)
end

function energy_bins(binfunc::Function, wbands::Vector{<:WannierBand}, E_range,
                     normalize_bins = false)
    nbins = length(E_range) - 1
    bins  = zeros(typeof(binfunc(wbands[1].eigvec[1])), nbins)
    # sum = zero(typeof(binfunc(bands[1].eigvec[1])))
    nperbin = zeros(Int, nbins)
    for b in wbands
        for (e, v) in zip(b.eigvals, b.eigvec)
            ie = findfirst(i -> E_range[i] <= e <= E_range[i+1], 1:nbins)
            if ie === nothing
                continue
            end
            bins[ie] += binfunc(v)
            nperbin[ie] += 1
        end
    end
    if normalize_bins #like taking the mean
        for i in 1:nbins
            if nperbin[i] > 0
                bins[i] /= nperbin[i]
            end
        end
    end
    return bins
end

function character_contribution(wband::WannierBand, atoms::Vector{Atom})
    contributions = zeros(length(wband.kpoints_cryst))
    for (i, v) in enumerate(wband.eigvec)
        for a in atoms
            contributions[i] += norm(v[a])^2
        end
    end
    return contributions
end

function DFControl.FileIO.pdos(wbands::Vector{<:WannierBand}, atoms::Vector{Atom},
                               dE = 0.02)
    Emin = minimum(wbands[1].eigvals)
    Emax = maximum(wbands[end].eigvals)
    E_range = range(Emin, Emax; step = dE)

    bins = energy_bins(wbands, E_range, false) do v
        tot = 0.0
        for a in atoms
            tot += norm(v[a])^2 / dE
        end
        return tot
    end
    return (E = E_range, pdos = bins ./ length(wbands[1].kpoints_cryst))
end

function kpdos(bands::Vector{<:WannierBand}, atoms::Vector{Atom})
    return map(x -> character_contribution(x, atoms), bands)
end
