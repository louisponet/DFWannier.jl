
struct BerryRGrid{T, MT, MT1}
    hami::TbHami{T, MT}
    A ::Vector{Vec3{MT}} #A_a(R) = <0|r_a|R> is the Fourier transform of the Berrry connection A_a(k) = i<u|del_a u> (a=x,y,z)the berry connection 
    B ::Vector{Vec3{MT}} #B_a(R)=<0n|H(r-R)|Rm> is the Fourier transform of B_a(k) = i<u|H|del_a u> (a=x,y,z)
    C ::Vector{MT1} #CC_ab(R) = <0|r_a.H.(r-R)_b|R> is the Fourier transform of CC_ab(k) = <del_a u|H|del_b u> (a,b=x,y,z)}
end

#A_a(R) = <0|r_a|R> is the Fourier transform of the Berrry connection A_a(k) = i<u|del_a u> (a=x,y,z)the berry connection
#B_a(R)=<0n|H(r-R)|Rm> is the Fourier transform of B_a(k) = i<u|H|del_a u> (a=x,y,z)
function BerryRGrid(ab_initio_grid::AbInitioKGrid{T}, hami::TbHami) where {T}
    irvec = map(x->x.R_cryst, hami)

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
            h    = kpoint.hamis[n]
            for v in 1:3
                t_fac = 1im * vr[v] * weight
                A_q[i][v] .+= t_fac .* overlap
                B_q[i][v] .+= t_fac .* h
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
    return BerryRGrid(hami, A_R, B_R, C_R)
end


struct BerryKGrid{T, MT, MT1} <: AbstractKGrid{T}
    hamiltonian_kgrid::HamiltonianKGrid{T, MT}
    J_plus ::Vector{Vec3{MT}}    
    J_minus::Vector{Vec3{MT}}
    A::Vector{Vec3{MT}}
    Ω::Vector{Vec3{MT}} #pseudo form of berry connection A
    B::Vector{Vec3{MT}}
    C::Vector{MT1}
    f::Vector{MT}
    g::Vector{MT}
end
function BerryKGrid(berry_R_grid::BerryRGrid, kpoints::Vector{<:Vec3}, fermi::AbstractFloat)
    tb_hami = berry_R_grid.hami
    hamiltonian_kgrid = HamiltonianKGrid(tb_hami, kpoints)
    nk = length(kpoints)
    J_plus  = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    J_minus = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    A_k = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    Ω_k = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    B_k = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    C_k = [Mat3([zeros_block(tb_hami) for j = 1:9]...) for i=1:nk]
    f = [zeros_block(tb_hami) for i=1:nk]
    g = [zeros_block(tb_hami) for i=1:nk]
    for i=1:nk
        Uk = hamiltonian_kgrid.eigvecs[i]
        Ek = hamiltonian_kgrid.eigvals[i]
        ∇Hk = Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami))
        A = A_k[i]
        B = B_k[i]
        Ω = Ω_k[i]
        C = C_k[i]
        fourier_transform(tb_hami, kpoints[i]) do n, iR, R_cart, b, fac
            Rcart = ustrip.(R_cart)
            for v=1:3
                ∇Hk[v][n] += Rcart[v] * 1im * fac * block(b)[n]
                A[v][n]   += fac * berry_R_grid.A[iR][v][n]

                B[v][n]   += fac * berry_R_grid.B[iR][v][n]
                for v2 = 1:3
                    C[v, v2][n] += fac * berry_R_grid.C[iR][v, v2][n]
                end
            end
            Ω[1][n] += 1im * fac * (Rcart[2] * berry_R_grid.A[iR][3][n] - Rcart[3] * berry_R_grid.A[iR][2][n])
            Ω[2][n] += 1im * fac * (Rcart[3] * berry_R_grid.A[iR][1][n] - Rcart[1] * berry_R_grid.A[iR][3][n])
            Ω[3][n] += 1im * fac * (Rcart[1] * berry_R_grid.A[iR][2][n] - Rcart[2] * berry_R_grid.A[iR][1][n])
        end
        J_plus_k  = J_plus[i]
        J_minus_k = J_minus[i]
        occupations_H_gauge = map(x -> x < fermi ? 1 : 0, Ek) #acting like it's only an insulator for now
        n_wann = length(Ek)
        
        f[i] .= Uk * diagm(0=>occupations_H_gauge) * Uk'
        # for n in 1:n_wann, m in 1:n_wann, j in 1:n_wann
        #     f[i][m, n] += Uk[m, j] * occupations_H_gauge[j] * conj(Uk[j, n])
        # end

        g[i] .= map(x -> -x, f[i])
        for j = 1:n_wann
            g[i][j, j] += 1
        end
        # @show g
        for v = 1:3
            Hbar = Uk' * ∇Hk[v] * Uk
            for m = 1:n_wann
                Ek_m = Ek[m]
                for n = 1:n_wann
                    Ek_n = Ek[n]
                    if Ek_n > fermi && Ek_m < fermi
                        J_plus_k[v][n, m]  =  1im * Hbar[n, m]/(Ek[m] - Ek[n])
                        J_minus_k[v][m, n] =  1im * Hbar[m, n]/(Ek[n] - Ek[m])
                        #else is already taken care of by initializing with zeros
                    end
                end
            end
            J_plus_k[v]  .= Uk * J_plus_k[v] * Uk'
            J_minus_k[v] .= Uk * J_plus_k[v] * Uk'
        end
    end
    return BerryKGrid(hamiltonian_kgrid, J_plus, J_minus, A_k, Ω_k, B_k, C_k, f, g)
end

function fourier_q_to_R(f::Function, q_vectors, R_vectors)
    for iR in 1:length(R_vectors)
        for ik in 1:length(q_vectors)
            phase = exp(-2im * π * (k_cryst(q_vectors[ik]) ⋅ R_vectors[iR]))
            f(iR, ik, phase)
        end
    end
end



function orbital_angular_momentum()
end

