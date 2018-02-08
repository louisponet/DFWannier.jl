#TODO a lot of optimization possible by splitting it up!
function calc_observables(structure::WanStructure{T}, kpoints::Vector{<:AbstractVector{T}}, soc=false) where T
    nat = length(structure.atoms)
    calc_angmoms!(structure)
    Sx, Sy, Sz = calc_spins(structure)

    L_k = Vector{Tuple{WanAtom{T}, Vector{Vec3{T}}}}[]
    S_k = Vector{Tuple{WanAtom{T}, Vector{Vec3{T}}}}[]
    eigvals_k = Vector{T}[]
    cm_k = Vector{Point3D{T}}[]
    for j=1:size(kpoints)[1]
        k = kpoints[j]
        dips = calc_k_dips(structure.tbdip, k)
        if soc
            hami = construct_soc_hami(hami_from_k(structure.tbhami, k), structure)
        else
            hami = hami_from_k(structure.tbhami, k)
        end
        eigvals, eigvecs = sorted_eig(hami)
        push!(eigvals_k, eigvals)
        L_t, S_t = eigangmomspin(eigvecs, structure.atoms, Sx, Sy, Sz)
        push!(L_k, L_t)
        push!(S_k, S_t)
        push!(cm_k, eigcm(dips, eigvecs))
    end
    return eigvals_k, L_k, S_k, cm_k
end

function eigangmomspin(eigvecs, atoms::Vector{WanAtom{T}}, Sx, Sy, Sz) where T
    outL = Tuple{WanAtom, Vector{Vec3{T}}}[]
    outS = Tuple{WanAtom, Vector{Vec3{T}}}[]
    len = size(eigvecs)[1]
    for (a, at) in enumerate(atoms)
        L_t    = Vec3{T}[]
        S_t    = Vec3{T}[]
        for i = 1:len
            eigvec = eigvecs[:, i]
            L = zero(Vec3{Complex{T}})
            S = zero(Vec3{Complex{T}})
            for i1 = 1:len
                at_index1 = wfc2atindex(atoms, i1)
                at_index1 == a && continue
                c1 = eigvec[i1]
                for i2 = 1:len
                    at_index2 = wfc2atindex(atoms, i2)
                    at_index1 == a && continue

                    at = atoms[at_index1]
                    c2 = eigvec[i2]

                    wfi1 = wfc2atwfcindex(atoms, i1)
                    wfi2 = wfc2atwfcindex(atoms, i2)
                    L += conj(c1) * c2 * at.angmom[wfi1,wfi2]
                    S += conj(c1) * c2 * Vec3(Sx[wfi1, wfi2], Sy[wfi1, wfi2], Sz[wfi1, wfi2])
                end
            end
            push!(L_t, real(L))
            push!(S_t, real(S))
        end
        push!(outL, (at, L_t))
        push!(outS, (at, S_t))
    end
    return outL, outS
end

function eigcm(dip::AbstractMatrix{Point3{T}}, eigvecs) where T
    out = Point3{T}[]
    for i = 1:size(eigvecs)[1]
        eigvec = eigvecs[:, i]
        en = enumerate(eigvec)
        d = zero(Point3{Complex{T}})
        for (i1, v1) in en, (i2, v2) in en
            d += conj(v1) * v2 * dip[i1, i2]
        end
        push!(out, real(d))
    end
    return out
end



@inline function wfc2atindex(atoms::Vector{<:WanAtom}, wfci)
    wfcounter = 0
    i = 1
    while true
        wfcounter += length(atoms[i].wfcs)
        if wfcounter >= wfci
            return mod1(i, length(atoms))
        else
            i+=1
            i = mod1(i, length(atoms))
        end
    end
end

@inline function wfc2atwfcindex(atoms::Vector{<:WanAtom}, wfci)
    i = 1
    while true
        at = atoms[i]
        if wfci - length(at.wfcs) < 1
            return wfci
        else
            wfci -= length(at.wfcs)
            i += 1
            i = mod1(i, length(atoms))
        end
    end
end
