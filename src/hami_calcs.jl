
function Hk!(out::AbstractMatrix{T}, tbhami, kpoint) where T
    fill!(out, zero(T))
    for block in tbhami
        out .+= e^(-2im*pi*(block.Rtpiba ⋅ kpoint)) .* block.block
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

# "Symmetrizes the hamiltonian such that it is fully periodic"
# function symmetrize!(tb_hami::Vector{TbBlock{T}}, structure::AbstractStructure{T}) where  T
#     centerh = getfirst(x->x.Rtpiba == Vec3(0,0,0), tb_hami).block
#     bonds_ = bonds(structure)
#     max = length(bonds_)
#     shbonds = [Bond(bond.at1, bond.at2, zero(T)) for bond in bonds_]
#     atoms  = structure.atoms
#     Threads.@threads for i=1:length(tb_hami)
#         block = tb_hami[i]
#         block.Rtpiba == Vec3(0,0,0) && continue
#         shiftedbonds!(shbonds, block.Rtpiba, structure)
#         for b in bonds_, shb in shbonds
#             if b == shb
#                 for (proj1, proj3) in zip(b.at1.projections, shb.at1.projections), (proj2, proj4) in zip(b.at2.projections, shb.at2.projections)
#                     for (r1,r3) in zip(range(proj1),range(proj3)), (r2,r4) in zip(range(proj2),range(proj4))
#                         @inbounds block.block[r3, r4] = centerh[r1, r2]
#                     end
#                 end
#             end
#         end
#     end
# end


function isAFMperiodic(at1, at2, at3, at4, R)
    pos1 = at1.position + R/2
    pos2 = at2.position - R/2
    return (norm(at3.position-pos1) < 1.0e-7&& at3.element.symbol == at1.element.symbol && norm(at4.position-pos2) < 1.0e-7 && at4.element.symbol == at2.element.symbol)
    # return ( at3.element.symbol == at1.element.symbol &&  at4.element.symbol == at2.element.symbol)
end

"Symmetrizes the hamiltonian such that it is fully periodic"
function symmetrize!(tb_hamis::NTuple{2, Vector{TbBlock{T}}}, structure::AbstractStructure{T}) where  T
    atoms = structure.atoms
    cell = structure.cell
    counter = 0
    for j=1:2
    counter = 0
        hami = tb_hamis[j]
        centerh = getfirst(x->x.Rtpiba == Vec3(0,0,0), tb_hamis[3-j]).block
        for i=1:length(hami)
            hb = hami[i]
            H = hb.block
            norm(hb.Rtpiba) != 1 && continue
            oppositeh = getfirst(x->x.Rtpiba == -hb.Rtpiba, tb_hamis[j]).block
            R = cell' * hb.Rtpiba
            # for (a1, at1) in enumerate(atoms), (a2, at2) in enumerate(atoms), (a3, at3) in enumerate(atoms), (a4,at4) in enumerate(atoms)
            for at1 in atoms, at2 in atoms, at3 in atoms, at4 in atoms
                if isAFMperiodic(at1, at2, at3, at4, R)
                    # println("id1: $a1, id2: $a2, id3: $a3, id4:$a4, R: $(hb.Rtpiba)")
                    for proj1 in at1.projections, proj2 in at2.projections, proj3 in at3.projections, proj4 in at4.projections
                        for (r1, r3) in zip(range(proj1), range(proj3)), (r2, r4) in zip(range(proj2),range(proj4))
                            centerh[r2,r1]= (centerh[r2,r1] + H[r4, r3])/2
                            centerh[r1,r2] = conj(centerh[r2,r1])
                            H[r4,r3] = centerh[r2,r1]
                            oppositeh[r3, r4] = conj(H[r4,r3])
                        end
                    end
                end
            end
        end
    end
end
