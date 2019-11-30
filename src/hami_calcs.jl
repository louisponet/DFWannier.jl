div1(x, y) = div(x - 1, y) + 1

w_eachindex(m::Matrix) = eachindex(m)

Hk_sum!(out::M, m::M, fac::T) where {T, M <: AbstractMatrix{T}} =
    out .+= fac .* m

function Hk!(out::M, tbhami::TbHami{T, M}, kpoint::Vec3{T}) where {T, M <: AbstractMatrix{Complex{T}}}
    fill!(out, zero(Complex{T}))
    for b in tbhami
	    fac = ℯ^(-2im*pi*(b.R_cryst ⋅ kpoint))
        Hk_sum!(out, block(b), fac)
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
