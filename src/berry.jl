struct BerryKGrid{T, MT} <: AbstractKGrid{T}
    hamiltonian_kgrid::HamiltonianKGrid{T, MT}
    J_plus ::Vector{Vec3{MT}}    
    J_minus::Vector{Vec3{MT}}    
end

function BerryKGrid(tb_hami::TbHami, kpoints::Vector{<:Vec3}, fermi::AbstractFloat)
    hamiltonian_kgrid = HamiltonianKGrid(tb_hami, kpoints)
    nk = length(kpoints)
    J_plus  = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    J_minus = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    for i=1:nk
        Uk = hamiltonian_kgrid.eigvecs[i]
        Ek = hamiltonian_kgrid.eigvals[i]
        ∇Hk = Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami))
        fourier_transform(tb_hami, kpoints[i]) do i, b, fac
            for v=1:3
                ∇Hk[v][i] += ustrip(b.R_cart[v]) * 1im * fac * block(b)[i]
            end
        end
        J_plus_k  = J_plus[i]
        J_minus_k = J_minus[i]
        for v = 1:3
            Hbar = Uk' * ∇Hk[v] * Uk
            for m = 1:length(Ek)
                Ek_m = Ek[m]
                for n = 1:length(Ek)
                    Ek_n = Ek[n]
                    if Ek_n > fermi && Ek_m < fermi
                        J_plus_k[v][n, m]  =  1im * Hbar[n, m]/(Ek[m] - Ek[n])
                        J_minus_k[v][m, n] =  1im * Hbar[m, n]/(Ek[m] - Ek[n])
                        #else is already taken care of by initializing with zeros
                    end
                end
            end
            J_plus_k[v]  .= Uk * J_plus_k[v] * Uk'
            J_minus_k[v] .= Uk * J_plus_k[v] * Uk'
        end
    end
    return BerryKGrid(hamiltonian_kgrid, J_plus, J_minus)
end

function fourier_q_to_R(f::Function, q_vectors, R_vectors)
    for iR in 1:length(R_vectors)
        for ik in 1:length(q_vectors)
            phase = exp(-2im * π * (k_cryst(q_vectors[ik]) ⋅ R_vectors[iR]))
            f(iR, ik, phase)
        end
    end
end


#A_a(R) = <0|r_a|R> is the Fourier transform of the Berrry connection A_a(k) = i<u|del_a u> (a=x,y,z)the berry connection
#B_a(R)=<0n|H(r-R)|Rm> is the Fourier transform of B_a(k) = i<u|H|del_a u> (a=x,y,z)
function berry_matrices(ab_initio_grid::AbInitioKGrid{T}, irvec) where {T}
    n_wann = n_wannier_functions(ab_initio_grid)

    berry_vec = () -> Vec3([zeros(Complex{T}, n_wann, n_wann) for i = 1:3]...)
    berry_mat = () -> Mat3([zeros(Complex{T}, n_wann, n_wann) for i = 1:9]...)

    n_kpoints = length(ab_initio_grid)

    A_q = [berry_vec() for k=1:n_kpoints]
    B_q = [berry_vec() for k=1:n_kpoints]
    C_q = [berry_mat() for k=1:n_kpoints]

    n_nearest = n_nearest_neighbors(ab_initio_grid)
    neighbor_weights = ab_initio_grid.neighbor_weights 
    for i in 1:n_kpoints
        kpoint = ab_initio_grid.kpoints[i]
        for n in 1:n_nearest
            neighbor_bond = kpoint.neighbors[n]
            weight        = neighbor_weights[n]
            vr            = ustrip.(neighbor_bond.vr)
            overlap = kpoint.overlaps[n]
            hami    = kpoint.hamis[n]
            for v in 1:3
                t_fac = 1im * vr[v] * weight
                A_q[i][v] .+= t_fac .* overlap
                B_q[i][v] .+= t_fac .* hami
                for n2 in 1:n_nearest
                    weight2 = neighbor_weights[n2]
                    neighbor_bond2 = kpoint.neighbors[n2]
                    vr2 = ustrip.(neighbor_bond2.vr)
                    uHu = kpoint.uHu[n2, n]
                    for v2 in 1:3
                        t_fac2 = t_fac * -1im * weight2 * vr2[v2]
                        C_q[i][v, v2] .+= t_fac2 .* uHu
                    end
                end
            end
        end
        for v in 1:3
            A_q[i][v] .= (A_q[i][v] + A_q[i][v]')/2
            for v2 in 1:v
                C_q[i][v, v2] .= C_q[i][v2, v]'
            end
        end
    end
    A_R = [berry_vec() for k=1:length(irvec)]
    B_R = [berry_vec() for k=1:length(irvec)]
    C_R = [berry_mat() for k=1:length(irvec)]
    f_to_fourier = (iR, ik, phase) -> begin
        for v in 1:3
            A_R[iR][v] .+= phase .* A_q[ik][v] 
            B_R[iR][v] .+= phase .* B_q[ik][v]
            for v2 in 1:3
                C_R[iR][v2, v] .+= phase .* C_q[ik][v2, v]
            end
        end
    end
    fourier_q_to_R(f_to_fourier, ab_initio_grid.kpoints, irvec)
    for i in 1:length(A_R)
        for v=1:3
            A_R[i][v] ./= n_kpoints
            B_R[i][v] ./= n_kpoints
            for v2 in 1:3
                C_R[i][v, v2] ./= n_kpoints
            end
        end
    end
    return A_R, B_R, C_R
end





