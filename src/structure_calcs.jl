function calc_observables(structure::WanStructure{T}, kpoints::Vector{Vec3{T}}, soc::Bool=false) where T
    nat = length(atoms(structure))
    matdim = getwandim(structure)
    klen = length(kpoints)
    calc_angmoms!(structure)
    Sx, Sy, Sz = calc_spins(structure)
    outbands = soc ? wannierbands(2*matdim, kpoints) : wannierbands(matdim, kpoints)
    Threads.@threads for i=1:klen
        k = kpoints[i]
        t_hami, dips = hami_dip_from_k(structure.tbhami, structure.tbdip, k)

        hami = soc ? construct_soc_hami(t_hami, structure) : t_hami
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

function calc_observables(structure::WanStructure{T}, k_points, k_range::StepRangeLen, args...) where T
    mid = div(size(k_points)[1],2)+1
    beg = Int64(k_range[1])
    steps= div(size(k_range)[1],2)
    last = Int64(k_range[end])
    kxs = [range(k_points[beg][1], stop = k_points[mid][1], length = steps)..., k_points[mid][1], range(k_points[mid][1], stop = k_points[last][1], length = steps)[2:end]...]
    kys = [range(k_points[beg][2],stop = k_points[mid][2], length = steps)...,k_points[mid][2] ,range(k_points[mid][2], stop = k_points[last][2], length = steps)[2:end]...]
    kzs = [range(k_points[beg][3],stop = k_points[mid][3], length = steps)...,k_points[mid][3] ,range(k_points[mid][3],stop = k_points[last][3],length = steps)[2:end]...]
    kxs_t = [range(k_points[beg][1],stop = k_points[mid][1],length = steps*100)... ,range(k_points[mid][1],stop = k_points[last][1],length = steps*100)[2:end]...]
    kys_t = [range(k_points[beg][2],stop = k_points[mid][2],length = steps*100)... ,range(k_points[mid][2],stop = k_points[last][2],length = steps*100)[2:end]...]
    kzs_t = [range(k_points[beg][3],stop = k_points[mid][3],length = steps*100)... ,range(k_points[mid][3],stop = k_points[last][3],length = steps*100)[2:end]...]
    # kxs[div(length(kxs),2)]+=0.00001

    kxs[div(length(kxs),2)] = kxs_t[div(length(kxs_t),2)]
    kxs[div(length(kxs),2)+1] = kxs_t[div(length(kxs_t),2)+2]
    kys[div(length(kxs),2)] = kys_t[div(length(kxs_t),2)]
    kys[div(length(kxs),2)+1] = kys_t[div(length(kxs_t),2)+2]
    kzs[div(length(kxs),2)] = kzs_t[div(length(kxs_t),2)]
    kzs[div(length(kxs),2)+1] = kzs_t[div(length(kxs_t),2)+2]
    k_points = [[kxs[i],kys[i],kzs[i]] for i=1:length(kxs)]
    return calc_observables(structure, Vec3{T}.(k_points), args...)
end

function hami_dip_from_k(tbhami, tbdip, k::Vec3{T}) where T
    matdim = size(tbhami[1].block)
    outham = zeros(Complex{T}, matdim)
    outdip = zeros(Point3{Complex{T}}, matdim)
    for i = 1:length(tbhami)
        Rtpiba = tbhami[i].Rtpiba
        hb = tbhami[i].block
        db = tbdip[i].block
        factor =  ℯ^(-2im*pi*(Rtpiba ⋅ k))
        outham .+= factor .* hb
        outdip .+= factor .* db
    end
    outdip_ = real(outdip)
    return outham, [outdip_ fill(zero(Point3{T}), matdim);fill(zero(Point3{T}), matdim) outdip_]
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
                        L += conj(c1) * c2 * angmom(at)[wfi1, wfi2]::Vec3{Complex{T}}
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
        len = length(wfcs(atoms[i]))
        wfcounter += len
        if wfcounter >= wfci
            return i, div1(outi, length(atoms)), len - wfcounter + wfci
        else
            outi += 1
            i +=1
            i = mod1(i, length(atoms))
        end
    end
end
