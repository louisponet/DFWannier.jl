div1(x, y) = div(x - 1, y) + 1

w_eachindex(m::Matrix) = eachindex(m)

function Hk!(out::AbstractMatrix, tbhami::TbHami, kpoint::Vec3)
    fill!(out, zero(eltype(out)))
    fourier_transform(tbhami, kpoint) do i, b, fac
        @inbounds out[i] += fac * block(b)[i]
    end
end

function Hk(tbhami::TbHami, kpoint::Vec3)
    out = similar(tbhami[1].block)
    Hk!(out, tbhami, kpoint)
    return out
end

"Fourier transforms the tight binding hamiltonian and calls the R_function with the current index and the phase."
function fourier_transform(R_function::Function, tb_hami::TbHami{T}, kpoint::Vec3) where {T}
    for b in tb_hami
        degen = b.wigner_seitz_degeneracy
        shifts_used = 0
        for i in eachindex(block(b))
            n_shifts = b.wigner_seitz_nshifts[i]
            for is in 1:n_shifts
                shift = b.wigner_seitz_shifts[shifts_used + is]
                fac = ℯ^(2im*π*((b.R_cryst + shift) ⋅ kpoint))/(degen * n_shifts)
                R_function(i, b, fac)
            end
            shifts_used += n_shifts
        end
    end
end

"Fourier transforms the tight binding hamiltonian and calls the R_function with the current index and the phase."
function fourier_transform_nows(R_function::Function, tb_hami::TbHami{T}, kpoint::Vec3) where {T}
    for b in tb_hami
        degen = b.wigner_seitz_degeneracy
        fac = ℯ^(2im*π*(b.R_cryst ⋅ kpoint))/degen
        for i in eachindex(block(b))
            R_function(i, b, fac)
        end
    end
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
