#TODO a lot of optimization possible by splitting it up!
function calculate_observables(structure::WanStructure{T}, kpoints::Vector{<:AbstractVector{T}}, soc=false) where T
    matdims = get_mat_dims(structure)
    Lx, Ly, Lz = calculate_angmoms(structure)
    Sx, Sy, Sz = calculate_spins(getwfcs(structure))

    Lx_k = Vector{T}[]
    Ly_k = Vector{T}[]
    Lz_k = Vector{T}[]
    eigvals_k = Vector{T}[]
    cm_k = Vector{Point3D{T}}[]
    for j=1:size(kpoints)[1]
        k = kpoints[j]
        dips = calculate_k_dips(structure.tbdip, k)
        if soc
            hami = construct_soc_hami(hami_from_k(structure.tbhami, k), Lx, Ly, Lz, structure)
        else
            hami = hami_from_k(structure.tbhami, k)
        end
        eigvals, eigvecs = sorted_eig(hami)
        push!(eigvals_k, eigvals)
        Lx_kt = T[]
        Ly_kt = T[]
        Lz_kt = T[]
        cm_kt = Point3D{T}[]
        for i=1:size(eigvals)[1]
            Lx_t = zero(T)
            Ly_t = zero(T)
            Lz_t = zero(T)
            cm_t = zero(SVector{3, Complex{T}})
            eigval = eigvals[i]
            eigvec = eigvecs[:, i]
            for i1=1:size(eigvec)[1]
                c1 = eigvec[i1]
                for i2=1:size(eigvec)[1]
                    c2 = eigvec[i2]
                    Lx_t += conj(c1) * c2 * Lx[i1, i2]
                    Ly_t += conj(c1) * c2 * Ly[i1, i2]
                    Lz_t += conj(c1) * c2 * Lz[i1, i2]
                    cm_t += conj(c1) * c2 * dips[i1, i2]
                end
            end
            push!(Lx_kt, real(Lx_t))
            push!(Ly_kt, real(Ly_t))
            push!(Lz_kt, real(Lz_t))
            push!(cm_kt, real(cm_t))
        end
        push!(Lx_k, Lx_kt)
        push!(Ly_k, Ly_kt)
        push!(Lz_k, Lz_kt)
        push!(cm_k, cm_kt)
    end
    return Lx_k, Ly_k, Lz_k, cm_k, eigvals_k
end

function calculate_observables_soc(structure::WanStructure{T}, kpoints::Vector{<:AbstractVector{T}}) where T
    matdims = get_mat_dims(structure)
    Lx, Ly, Lz = calculate_angmoms(structure)
    Sx, Sy, Sz = calculate_spins(getwfcs(structure))

    Lx_k = Vector{T}[]
    Ly_k = Vector{T}[]
    Lz_k = Vector{T}[]
    cm_k = Vector{Point3D{T}}[]
    for j=1:size(kpoints)[1]
        k = kpoints[j]
        dips = calculate_k_dips(structure.tbdip, k)
        hami =
        eigvals, eigvecs = sorted_eig(hami)
        Lx_kt = T[]
        Ly_kt = T[]
        Lz_kt = T[]
        cm_kt = Point3D{T}[]
        for i=1:size(eigvals)[1]
            Lx_t = zero(T)
            Ly_t = zero(T)
            Lz_t = zero(T)
            cm_t = zero(SVector{3, Complex{T}})
            eigval = eigvals[i]
            eigvec = eigvecs[:, i]
            for i1=1:size(eigvec)[1]
                c1 = eigvec[i1]
                for i2=1:size(eigvec)[1]
                    c2 = eigvec[i2]
                    Lx_t += conj(c1) * c2 * Lx[i1, i2]
                    Ly_t += conj(c1) * c2 * Ly[i1, i2]
                    Lz_t += conj(c1) * c2 * Lz[i1, i2]
                    cm_t += conj(c1) * c2 * dips[i1, i2]
                end
            end
            push!(Lx_kt, real(Lx_t))
            push!(Ly_kt, real(Ly_t))
            push!(Lz_kt, real(Lz_t))
            push!(cm_kt, real(cm_t))
        end
        push!(Lx_k, Lx_kt)
        push!(Ly_k, Ly_kt)
        push!(Lz_k, Lz_kt)
        push!(cm_k, cm_kt)
    end
    return Lx_k, Ly_k, Lz_k, cm_k
end
