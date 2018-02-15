function calc_observables(structure::WanStructure{T}, kpoints::Vector{<:AbstractVector{T}}, soc::Bool=false) where T
    nat = length(structure.atoms)
    matdim = getwandim(structure)
    klen = length(kpoints)
    calc_angmoms!(structure)
    Sx, Sy, Sz = calc_spins(structure)
    outbands = WannierBand{T}[]
    if soc
        for i =1:2*matdim
            push!(outbands, WannierBand(kpoints))
        end
    else
        for i =1:matdim
            push!(outbands, WannierBand(kpoints))
        end
    end

    Threads.@threads for i=1:klen
        k = kpoints[i]
        t_hami, dips = hami_dip_from_k(structure.tbhami, structure.tbdip, k)
        if soc
            hami = construct_soc_hami(t_hami, structure)
        else
            hami = t_hami
        end
        eigvals, eigvecs = sorted_eig(hami)
        eigvals_k = real(eigvals)
        cm_k      = eigcm(dips, eigvecs)
        L_k, S_k  = eigangmomspin(eigvecs, structure.atoms, Sx, Sy, Sz)
        for j = 1:length(eigvals)
            ob = outbands[j]
            ob.eigvals[i] = eigvals_k[j]
            ob.eigvec[i]  = eigvecs[:, j]
            ob.angmoms[i] = [l[j] for l in L_k]
            ob.cms[i]     = cm_k[j]
            ob.spins[i]   = [s[j] for s in S_k]
        end
    end

    return outbands
end

function calc_observables(structure::WanStructure, k_points, k_range::StepRangeLen, args...)
    mid = div(size(k_points)[1],2)+1
    beg = Int64(k_range[1])
    steps = div(size(k_range)[1],2)
    last = Int64(k_range[end])
    kxs = [linspace(k_points[beg][1], k_points[mid][1], steps)..., k_points[mid][1], linspace(k_points[mid][1], k_points[last][1], steps)[2:end]...]
    kys = [linspace(k_points[beg][2],k_points[mid][2],steps)...,k_points[mid][2] ,linspace(k_points[mid][2], k_points[last][2], steps)[2:end]...]
    kzs = [linspace(k_points[beg][3],k_points[mid][3],steps)...,k_points[mid][3] ,linspace(k_points[mid][3],k_points[last][3],steps)[2:end]...]
    kxs_t = [linspace(k_points[beg][1],k_points[mid][1],steps*100)... ,linspace(k_points[mid][1],k_points[last][1],steps*100)[2:end]...]
    kys_t = [linspace(k_points[beg][2],k_points[mid][2],steps*100)... ,linspace(k_points[mid][2],k_points[last][2],steps*100)[2:end]...]
    kzs_t = [linspace(k_points[beg][3],k_points[mid][3],steps*100)... ,linspace(k_points[mid][3],k_points[last][3],steps*100)[2:end]...]
    # kxs[div(length(kxs),2)]+=0.00001
    kxs[div(length(kxs),2)] = kxs_t[div(length(kxs_t),2)]
    kxs[div(length(kxs),2)+1] = kxs_t[div(length(kxs_t),2)+2]
    kys[div(length(kxs),2)] = kys_t[div(length(kxs_t),2)]
    kys[div(length(kxs),2)+1] = kys_t[div(length(kxs_t),2)+2]
    kzs[div(length(kxs),2)] = kzs_t[div(length(kxs_t),2)]
    kzs[div(length(kxs),2)+1] = kzs_t[div(length(kxs_t),2)+2]
    k_points = [[kxs[i],kys[i],kzs[i]] for i=1:length(kxs)]
    return calc_observables(structure, k_points, args...)
end

function hami_dip_from_k(tbhami, tbdip, k::Vector{T}) where T
    dim = 0
    for i = 1:length(tbhami)
        d = tbhami[i][4]
        if d > dim
            dim = d
        else
            break
        end
    end
    outhami = zeros(Complex{T},(dim,dim))
    outdip  = zeros(Point3{Complex{T}},(dim,dim))
    for i = 1:size(tbhami)[1]
        h = tbhami[i]
        a, b, c = h[1], h[2], h[3]
        wf1, wf2 = h[4], h[5]
        exponent =  2pi * (k[1]*a + k[2]*b + k[3]*c)
        factor   = exp(-1im * exponent)
        if wf2 == wf1
            outhami[wf1, wf2] += h[6] * cos(exponent)
        else
            outhami[wf1, wf2] += h[6] * factor
        end
        outdip[wf1, wf2] += tbdip[i][6] * factor
    end
    outdip_ = real(outdip)
    return Hermitian(outhami), [outdip_ zeros(outdip_);zeros(outdip_) outdip_]
end

function eigangmomspin(eigvecs, atoms::Vector{WanAtom{T}}, Sx, Sy, Sz) where T
    outL = Vector{Vec3{T}}[]
    outS = Vector{Vec3{T}}[]
    len = size(eigvecs)[1]
    for (a, at) in enumerate(atoms)
        L_t    = Vec3{T}[]
        S_t    = Vec3{T}[]
        for i = 1:len
            eigvec = eigvecs[:, i]
            L = zero(Vec3{Complex{T}})
            S = zero(Vec3{Complex{T}})
            for i1 = 1:len
                at_index1, blockid1, wfi1 = wfc2atindex(atoms, i1)
                at_index1 != a && continue
                c1 = eigvec[i1]
                for i2 = 1:len
                    at_index2, blockid2, wfi2= wfc2atindex(atoms, i2)
                    at_index2 != a && continue
                    at = atoms[at_index1]
                    c2 = eigvec[i2]

                    if blockid1 == blockid2
                        L += conj(c1) * c2 * at.angmom[wfi1, wfi2]::Vec3{Complex{T}}
                    else
                        S += conj(c1) * c2 * Vec3(Sx[i1, i2], Sy[i1, i2], Sz[i1, i2])
                    end
                end
            end
            push!(L_t, real(L))
            push!(S_t, real(S))
        end
        push!(outL, L_t)
        push!(outS, S_t)
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

function wfc2atindex(atoms::Vector{<:WanAtom}, wfci)
    wfcounter = 0
    i = 1
    outi = 1
    while true
        len = length(atoms[i].wfcs)
        wfcounter += length(atoms[i].wfcs)
        if wfcounter >= wfci
            return i, div1(outi, length(atoms)), len - wfcounter + wfci
        else
            outi += 1
            i +=1
            i = mod1(i, length(atoms))
        end
    end
end
