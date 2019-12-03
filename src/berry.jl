struct BerryKGrid{T, MT} <: AbstractKGrid{T}
    hamiltonian_kgrid::HamiltonianKGrid{T, MT}
    J_plus ::Vector{Vec3{MT}}    
    J_minus::Vector{Vec3{MT}}    
end

function BerryKGrid(tb_hami::TbHami, kpoints::Vector{<:Vec3})
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
        for v = 1:3
            Hbar = Uk' * ∇Hk[v] * Uk
            # Hbar = ∇Hk[v]
            for m = 1:length(Ek)
                Ek_m = Ek[m]
                for n = 1:length(Ek)
                    Ek_n = Ek[n]
                    if 
                    D[i][v][n, m] = n == m ? 0 : Hbar[n,m]/(Ek[m]-Ek[n])
                end
            end
        end
    end
    return BerryKGrid(hamiltonian_kgrid, D)
end
