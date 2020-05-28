
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
    # @show A_q[1][1]
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
    J::Vector{Vec3{MT}}
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
    J = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    A_k = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    Ω_k = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    B_k = [Vec3(zeros_block(tb_hami), zeros_block(tb_hami), zeros_block(tb_hami)) for i=1:nk]
    C_k = [Mat3([zeros_block(tb_hami) for j = 1:9]...) for i=1:nk]
    f = [zeros_block(tb_hami) for i=1:nk]
    g = [zeros_block(tb_hami) for i=1:nk]
    Threads.@threads for i=1:nk
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
        J_k = J[i]
        occupations_H_gauge = map(x -> x < fermi ? 1 : 0, Ek) #acting like it's only an insulator for now
        n_wann = length(Ek)
        
        f[i] .= Uk * diagm(0 => occupations_H_gauge) * Uk'

        g[i] .= map(x -> -x, f[i])
        for j = 1:n_wann
            g[i][j, j] += 1
        end
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
                    J_k[v][m, n] =  n==m ? 0.0 : 1im * Hbar[m, n]/(Ek[n] - Ek[m]) 
                end
            end
            J_plus_k[v]  .= Uk * J_plus_k[v] * Uk'
            J_minus_k[v] .= Uk * J_minus_k[v] * Uk'
            J_k[v] .= Uk * J_k[v] * Uk'
        end
    end
    return BerryKGrid(hamiltonian_kgrid, J_plus, J_minus, J, A_k, Ω_k, B_k, C_k, f, g)
end

n_wannier_functions(bgrid::BerryKGrid) = size(bgrid.f[1], 1)
n_kpoints(bgrid::BerryKGrid)           = length(bgrid.f)

core_kgrid(bgrid::BerryKGrid) = core_kgrid(bgrid.hamiltonian_kgrid)

for f in (:Hk, :eigvecs, :eigvals)
    @eval $f(bgrid::BerryKGrid) = $f(bgrid.hamiltonian_kgrid)
end


function fourier_q_to_R(f::Function, q_vectors, R_vectors)
    for iR in 1:length(R_vectors)
        for ik in 1:length(q_vectors)
            phase = exp(-2im * π * (k_cryst(q_vectors[ik]) ⋅ R_vectors[iR]))
            f(iR, ik, phase)
        end
    end
end

const pseudo_α = Vec3(2,3,1)
const pseudo_β = Vec3(3,1,2)


function orbital_angular_momentum(berry_K_grid::BerryKGrid{T}, fermi) where {T}
    nwann = n_wannier_functions(berry_K_grid)
    nk = n_kpoints(berry_K_grid)

    M_local     = [Vec3([zeros(T, nwann, nwann) for i =1:3]...) for ik=1:nk]
    M_itinerant = [Vec3([zeros(T, nwann, nwann) for i =1:3]...) for ik=1:nk]

    Threads.@threads for ik in 1:nk
        f = berry_K_grid.f[ik]
        g = berry_K_grid.g[ik]
        A = berry_K_grid.A[ik] 
        B = berry_K_grid.B[ik] 
        H = berry_K_grid.hamiltonian_kgrid.Hk[ik]
        C = berry_K_grid.C[ik]
        J = berry_K_grid.J[ik]
        Ω = berry_K_grid.Ω[ik]
        for iv in 1:3
            α = pseudo_α[iv]
            β = pseudo_β[iv]

            Fαβ = real.(f * Ω[iv]) .- 2 .* imag.(f * A[α] * g * J[β] .+ f * J[α] * g * A[β] .+ f * J[α] * g * J[β])

            Hαβ = real.(f * H * f * Ω[iv]) .+ 2 .* imag.(f * H * f * A[α] * f * A[β] .- f * H * f * (A[α] * g * J[β] + J[α] * g * A[β] + J[α] * g * J[β]))

            J0 = real.(f * -1im*(C[α, β] - C[α, β]')) .- 2 .* imag.(f * H * A[α] * f * A[β])
            J1 = - 2 .* imag.(f * J[α] * g * B[β] .- f * J[β] * g * B[α])
            J2 = - 2 .* imag.(f * J[α] * g * H * g * J[β])

            Gαβ = J0 + J1 + J2

            M_local[ik][iv] .= Gαβ .- fermi .* Fαβ
            M_itinerant[ik][iv] .= Hαβ .- fermi .* Fαβ
        end
    end
    return M_local, M_itinerant
end

function orbital_angular_momentum_w90(berry_K_grid::BerryKGrid{T}) where {T}
    nwann = n_wannier_functions(berry_K_grid)
    nk = n_kpoints(berry_K_grid)

    non_traced_Fαβ = [Vec3([zeros(T, nwann, nwann) for i =1:3]...) for ik=1:nk]
    non_traced_Hαβ = [Vec3([zeros(T, nwann, nwann) for i =1:3]...) for ik=1:nk]
    non_traced_Gαβ = [Vec3([zeros(T, nwann, nwann) for i =1:3]...) for ik=1:nk]

    for ik in 1:nk
    # for ik in 1:nk
        f = berry_K_grid.f[ik]
        g = berry_K_grid.g[ik]
        A = berry_K_grid.A[ik] 
        B = berry_K_grid.B[ik] 
        H = berry_K_grid.hamiltonian_kgrid.Hk[ik]
        C = berry_K_grid.C[ik]
        J_plus = map(x->x', berry_K_grid.J_plus[ik])
        J_minus = map(x->x', berry_K_grid.J_minus[ik])
        Ω = berry_K_grid.Ω[ik]
        for iv in 1:3
            α = pseudo_α[iv]
            β = pseudo_β[iv]

            non_traced_Fαβ[ik][iv] .+= real.(f * Ω[iv]) .- 2 .* imag.(A[α] * J_plus[β] .+ J_minus[α] * A[β] .+ J_minus[α] * J_plus[β])
            t_1 = H * A[α]
            t_3 = H * Ω[iv]

            t_2 = -1im*(C[α, β] - C[α, β]')

            t_4 = t_1 * f

            t_5 = t_4 * A[β]

            s = 2 .* imag.(f * t_5)

            non_traced_Gαβ[ik][iv] .+= real.(f * t_2) .- s
            non_traced_Hαβ[ik][iv] .+= real.(f * t_3) .+ s

            t_4 = H * J_minus[α]

            non_traced_Gαβ[ik][iv] .-= 2 .* imag.(J_minus[α] * B[β] .- J_minus[β] * B[α])

            non_traced_Hαβ[ik][iv] .-= 2 .* imag.(t_1 * J_plus[β] .+ t_4 * A[β])
            t_4 = J_minus[α] * H
            t_5 = H * J_minus[α]
            
            non_traced_Gαβ[ik][iv] .-= 2 .* imag.(t_4 * J_plus[β])
            
            non_traced_Hαβ[ik][iv] .-= 2 .* imag.(t_5 * J_plus[β])

        end
    end
    return non_traced_Fαβ, non_traced_Hαβ, non_traced_Gαβ
end
