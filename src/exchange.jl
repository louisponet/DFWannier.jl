using DFControl: Projection, Orbital, Structure, orbital, size, orbsize
"""
    WannExchanges{T <: AbstractFloat}

This holds the exhanges between different orbitals and calculated sites.
Projections and atom datablocks are to be found in the corresponding wannier input file.
It turns out the ordering is first projections, then atom order in the atoms datablock.
"""
mutable struct Exchange{T <: AbstractFloat}
    J       ::Matrix{T}
    atom1   ::Atom{T}
    atom2   ::Atom{T}
    proj1   ::Projection
    proj2   ::Projection
end

function DHvecvals(hamis, k_grid)
    Hvecs = [[similar(hami[1].block) for i=1:length(k_grid)] for hami in hamis]
    Hvals = [[similar(hami[1].block[:,1]) for i=1:length(k_grid)] for hami in hamis]
    D    = [zeros(eltype(hamis[1][1].block), size(hamis[1][1].block)) for i=1:Threads.nthreads()]
    Threads.@threads for i=1:length(k_grid)
        for j=1:2
            fac = (-1)^(j - 1)
            tid = Threads.threadid()
            #= Hvecs[j][i] is used as a temporary cache to store H(k) in. Since we
            don't need H(k) but only Hvecs etc, this is ok.
            =#
            Hk!(Hvecs[j][i], hamis[j], k_grid[i])
            D[tid] .+= fac .* Hvecs[j][i]
            Hvals[j][i], Hvecs[j][i] = LAPACK.syevr!('V', 'A', 'U', Hvecs[j][i], 0.0, 0.0, 0, 0, -1.0)
        end
    end
    return Hvecs, Hvals, sum(D)/length(k_grid)
end

function setup_exchanges(atoms::Vector{<:AbstractAtom{T}}, orbitals) where T <: AbstractFloat
    exchanges = Exchange{T}[]
    for (i, at1) in enumerate(atoms), at2 in atoms
        for proj1 in projections(at1), proj2 in projections(at2)
            if proj1.orb in orbitals && proj2.orb in orbitals
                push!(exchanges, Exchange{T}(zeros(T, orbsize(proj1), orbsize(proj1)), at1, at2, proj1, proj2))
            end
        end
    end
    return exchanges
end
function setup_ω_grid(ωh, ωv, n_ωh, n_ωv, offset=0.00)
    ω_grid = vcat(range(ωh, stop=ωh + ωv*1im, length=n_ωv)[1:end-1],
                  range(ωh + ωv*1im, stop=offset + ωv*1im, length=n_ωh)[1:end-1],
                  range(offset + ωv*1im, stop=offset, length=n_ωv))
    return ω_grid
end

function G!(G, cache1, cache2, cache3, ω::T, μ, Hvecs, Hvals, R, kgrid) where T
    dim = size(G)[1]
    fill!(G, zero(T))
    for ik=1:length(kgrid)
        fill!(cache1, zero(T))
        k_phase = exp(2im * π * dot(R,kgrid[ik]))
        for x=1:dim
            cache1[x, x] = 1.0 /(μ + ω - Hvals[ik][x])
        end
        @! cache2 = Hvecs[ik] * cache1
        adjoint!(cache3, Hvecs[ik])
        @! cache1 = cache2 * cache3

        G .+= cache1 .* k_phase
    end
end

#DON'T FORGET HAMIS ARE UP DOWN ORDERED!!!
function calcexchanges(hamis,  structure::Structure, fermi::T;
                             nk::NTuple{3, Int} = (10, 10, 10),
                             R                  = Vec3(0, 0, 0),
                             ωh::T              = T(-30.), #starting energy
                             ωv::T              = T(0.1), #height of vertical contour
                             n_ωh::Int          = 300,
                             n_ωv::Int          = 50,
                             temp::T            = T(0.01),
                             orbitals::Array{Symbol, 1} = [:d, :f]) where T <: AbstractFloat
    orbitals = orbital.(orbitals)
    @assert !all(isempty.(projections.(DFControl.atoms(structure)))) "Please read a valid wannier file for structure with projections."
    nth      = Threads.nthreads()
    μ        = fermi
    atoms    = structure.atoms
    k_grid   = [Vec3(kx, ky, kz) for kx = 0.5/nk[1]:1/nk[1]:1, ky = 0.5/nk[2]:1/nk[2]:1, kz = 0.5/nk[3]:1/nk[3]:1]

    Hvecs, Hvals, D = DHvecvals(hamis, k_grid)
    n_orb = size(D)[1]

    ω_grid    = setup_ω_grid(ωh, ωv, n_ωh, n_ωv)
    exchanges = setup_exchanges(atoms, orbitals)

    t_js                      = [[zeros(T, size(e.J)) for t=1:nth] for e in exchanges]
    caches1, caches2, caches3 = [[zeros(Complex{T}, n_orb, n_orb) for t=1:nth] for i=1:3]
    totocc_t                  = [zero(Complex{T}) for t=1:nth]
    gs                        = [[zeros(Complex{T}, n_orb, n_orb) for n=1:2] for t  =1:nth]
    # Threads.@threads for j=1:length(ω_grid[1:end-1])
    Threads.@threads for j=1:length(ω_grid[1:end-1])
        tid = Threads.threadid()
        ω   = ω_grid[j]
        dω  = ω_grid[j + 1] - ω
        g   = gs[tid]
        for s = 1:2
            R_ = (-1)^(s-1) * R #R for spin up (-1)^(0) == 1, -R for spin down
            G!(g[s], caches1[tid], caches2[tid], caches3[tid], ω, μ, Hvecs[s], Hvals[s], R_, k_grid)
        end
        for (eid, exch) in enumerate(exchanges)
            rm = range(exch.proj1)
            rn = range(exch.proj2)
            t_js[eid][tid] .+= sign(real(tr(view(D, rm, rm)))) .* sign(real(tr(view(D,rn, rn)))) .* imag(view(D,rm, rm) * view(g[1],rm, rn) * view(D,rn, rn) * view(g[2],rn, rm) * dω)
        end
    end
    for (eid, exch) in enumerate(exchanges)
        exch.J = 1e3 / (2π * length(k_grid)^2) * sum(t_js[eid])
    end
    structure.data[:totocc] = real(totocc(Hvals, fermi, temp))
    structure.data[:exchanges] = exchanges
end

function totocc(Hvals, fermi::T, temp::T) where T
    totocc = zero(Complex{T})
    for s=1:2
        for i = 1:length(Hvals[s])
            totocc += sum( 1 ./ (exp.((Hvals[s][i].-fermi)./temp) .+ 1))
        end
    end
    return totocc/length(Hvals[1])
end


function calcexchanges(hami_up_file::String, hami_down_file::String, wannier_input_file::String, args...; kwargs...)
    structure = read_wannier_input(wannier_input_file)[2]
    calcexchanges(symmetrize!(readhami(hami_up_file,structure), structure), symmetrize!(readhami(hami_down_file, structure), structure), structure, args...; kwargs...)
    return structure
end

function calcexchanges(hami_up, hami_down, wannier_input_file::String, args...; kwargs...)
    structure = read_wannier_input(wannier_input_file)[2]
    calcexchanges(hami_up, hami_down, structure, args...; kwargs...)
    return structure
end


@inline function Jmn_index(atom1::Int, atom2::Int, total::Int)
    index = -1
    for i = 1:atom1 - 1
        index += total - i - 1
    end
    index += atom2
    return index
end

function exchange_between(atom1::Atom{T}, atom2::Atom{T}, exchanges::Array{Exchange{T}, 1}) where T <: AbstractFloat
    exch = zero(T)
    for e in exchanges
        if e.atom1 == atom1 && e.atom2 == atom2
            exch += tr(e.J)
        end
    end
    return exch
end
