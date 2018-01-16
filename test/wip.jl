
using DFWannier

qe_2a   = ldsj("GdMn2O5/NSOC/qe/2a/", phd_dir*"GdMn2O5/NSOC/qe/2a/")

scfout = pull_outputs(qe_2a,extras=["1scf.out"])[1]
exch_file = pull_outputs(qe_2a,extras=["exchanges.txt"])[1]

infos = DFWannier.get_wan_projections(qe_2a.local_dir * "wan_up.win")
data  = read_qe_output(scfout)
atoms = Array{Atom{Float64},1}()
for i=1:length(data[:colin_mag_moments])
    moment = data[:colin_mag_moments][i]
    push!(atoms, Atom(infos[i].position, 0.0, [0., 0., moment]))
end

Jmn = read_exchanges(qe_2a.local_dir * "exchanges.txt")

function heisenberg_energy(atoms, exchanges, H)
    energy = zero(Float64)
    for i = 1:length(atoms)
        for j = i+1:length(atoms)
            energy += exchanges[i,j] * dot(atoms[i].mag_moment,atoms[j].mag_moment) + H * atoms[i].mag_moment[3]
        end
    end
    return energy
end

test = heisenberg_energy(atoms, Jmn, 025)