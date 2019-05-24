div1(x, y) = div(x - 1, y) + 1

function sorted_eigt(hami::Matrix)
    eigs, eigvec = (eigen(hami)...,)
    out_eig      = similar(eigs)
    out_vec      = similar(eigvec)
    perm         = sortperm(real(eigs))
    for (i, p) in enumerate(perm)
	    out_vec[:, i] = eigvec[:,p]
	    out_eig[i]   = eigs[p]
    end
    return out_eig, out_vec
end

function sorted_eig(m::AbstractMatrix{Complex{T}}) where {T}
	out_vecs = zeros(m)
	out_vals = zeros(T, size(m)[1])
    return sorted_eig!(out_vals, out_vecs, m)
end

function sorted_eig!(out_vals::Vector{T}, out_vecs::BlockBandedMatrix{Complex{T}}, m::BlockBandedMatrix{Complex{T}}) where {T}
	dim = size(m)[1]
	b_ranges = [1:div(dim, 2), div(dim, 2)+1:dim]
	for j=1:2
		b = Block(j,j)
	    @time out_vals[b_ranges[j]], out_vecs[b] = LAPACK.syevr!('V', 'A', 'U', m[b], 0.0, 0.0, 0, 0, -1.0)
    end
    return out_vals, out_vecs
end

sorted_eig!(out_vals::Vector{T}, out_vecs::Matrix{Complex{T}}, m::Matrix{Complex{T}}) where {T} =
	out_vals, out_vecs = LAPACK.syevr!('V', 'A', 'U', m, 0.0, 0.0, 0, 0, -1.0)


w_eachindex(m::Matrix) = eachindex(m)
w_eachindex(m::BlockBandedMatrix) = eachindex()

Hk_sum!(out::M, m::M, fac::T) where {T, M <: Matrix{T}} =
	out .+= fac .* m
Hk_sum!(out::M, m::M, fac::T) where {T, M <: BlockBandedMatrix{T}} =
    out.data .+= fac .* m.data

function Hk!(out::M, tbhami::TbHami{T, M}, kpoint::Vec3{T}) where {T, M <: AbstractMatrix{Complex{T}}}
    fill!(out, zero(Complex{T}))
    for b in tbhami
	    fac = ℯ^(-2im*pi*(b.R_cryst ⋅ kpoint))
        Hk_sum!(out, block(b), fac)
    end
    for i=1:size(out)[1]
        out[i,i] = real(out[i,i]) + 0.0im
    end
end

function Hk(tbhami, kpoint)
    out = similar(tbhami[1].block)
    Hk!(out, tbhami, kpoint)
    return out
end

"Constructs the total spin-orbit-coupled Hamiltonian out of supplied angular momentums between the Wannier functions and uses the l_soc of the atoms."
function construct_soc_hami(hami, structure::WanStructure{T})::Matrix{Complex{T}} where T
    dim = getwandim(structure)
    Lx_soc = zeros(Complex{T}, dim, dim)
    Ly_soc = zeros(Complex{T}, dim, dim)
    Lz_soc = zeros(Complex{T}, dim, dim)
    i = 1
    for at in atoms(structure)
        soc = lsoc(at)
        mom = angmom(at)
        len = length(wfcs(at))-1
        Lx_soc[i:i+len, i:i+len] = 0.5 * soc * getindex.(mom, 1)
        Ly_soc[i:i+len, i:i+len] = 0.5 * soc * getindex.(mom, 2)
        Lz_soc[i:i+len, i:i+len] = 0.5 * soc * getindex.(mom, 3)
        i += len + 1
    end
    Lx_soc = (Lx_soc+Lx_soc')/2
    Ly_soc = (Ly_soc+Ly_soc')/2
    Lz_soc = (Lz_soc+Lz_soc')/2
    out = [hami+Lz_soc Lx_soc-1im*Ly_soc;Lx_soc+1im*Ly_soc hami-Lz_soc]
    return out
end

function heisenberg_energy(moments::Vector{<:Vec3}, exchanges::Vector{Matrix{T}}, H, anisotropy) where T
    energy = zero(T)
    nexch  = length(exchanges)
    nmom   = length(moments)
    for i = 1:nmom
        for j = i+1:nmom
            for k = 1:nexch
                energy += exchanges[k][i,j] * dot(moments[i], moments[j]) + H * moments[i][1] + anisotropy * moments[i][3]^2
            end
        end
    end
    return energy
end

function AFMmap(structure, Rcryst)
    R = cell(structure)' * Rcryst/2
    atoms = atoms(structure)
    map1 = Dict{AbstractAtom, AbstractAtom}()
    for at1 in atoms, at2 in atoms
        if bondlength(at2, at1, R) < 1.0e-7
            map1[at1] = at2
        end
    end
    return map1
end

function rs(R, r1, r2, r3, r4)
    sig =
    if sign(R) < 0
        return r4, r3, r2, r1
    else
        return r1, r2, r3, r4
    end
end

function symmetrize!(tb_hamis::NTuple{2, Vector{TbBlock{T}}}, structure::AbstractStructure{T}) where  T
    forwardmap = AFMmap(structure, Vec3(1,0,0))
    Hup = getfirst(x -> x.R_cryst == Vec3(0,0,0), tb_hamis[1]).block
    Hdn = getfirst(x -> x.R_cryst == Vec3(0,0,0), tb_hamis[2]).block

    for (at1, at2) in forwardmap, (at3, at4) in forwardmap
        for (r1, r2) in zip(range.(projections(at1)), range.(projections(at2))), (r3, r4) in zip(range.(projections(at3)), range.(projections(at4)))
            Hup[r1, r3] .= (Hup[r1, r3] .+ Hdn[r2, r4]) ./ 2
            Hdn[r2, r4] .= Hup[r1, r3]

            for R=-1:2:1
                Hu = getfirst(x -> x.R_cryst == Vec3(R,0,0), tb_hamis[1]).block
                Hd = getfirst(x -> x.R_cryst == Vec3(R,0,0), tb_hamis[2]).block
                r1_, r2_, r3_, r4_ = rs(R, r1, r2, r3, r4)
                Hu[r2_, r3_] .= (Hu[r2_, r3_] .+ Hdn[r1_, r4_]) ./ 2
                Hdn[r1_, r4_] .= Hu[r2_, r3_]
                Hd[r2_, r3_]  .= (Hd[r2_, r3_] .+ Hup[r1_, r4_]) ./ 2
                Hup[r1_, r4_] .= Hd[r2_, r3_]
            end
        end
    end
end
