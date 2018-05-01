
function Hk!(out::Matrix{T}, tbhami, kpoint) where T
    fill!(out, zero(T))
    for block in tbhami
        out .+= e^(-2im*pi*(block.Rtpiba ⋅ kpoint)) .* block.block
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
    for at in structure.atoms
        len = length(at.wfcs)-1
        Lx_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 1)
        Ly_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 2)
        Lz_soc[i:i+len, i:i+len] = 0.5 * at.lsoc * getindex.(at.angmom, 3)
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

"Symmetrizes the hamiltonian such that it is fully periodic"
function symmetrize!(tb_hami, structure)
    centerh = getfirst(x->x.Rtpiba == Vec3(0,0,0), tb_hami).block
    bonds_ = bonds(structure)
    atoms  = structure.atoms
    for block in tb_hami
        block.Rtpiba == Vec3(0,0,0) && continue
        shbonds = shiftedbonds(block.Rtpiba, structure)
        for b in bonds_, shb in shbonds
            if b == shb
                for j1=1:length(b.at1.projections), j2=1:length(b.at2.projections)
                    block.block[range(shb.at1.projections[j1]), range(shb.at2.projections[j2])] .= centerh[range(b.at1.projections[j1]), range(b.at2.projections[j2])]
                end
            end
        end
    end
end
