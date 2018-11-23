import DFControl: bondlength, AbstractAtom
mutable struct Bond{T<:AbstractFloat}
    at1::AbstractAtom{T}
    at2::AbstractAtom{T}
    length::T
end

Base.length(bond::Bond) = bond.length

bonds(structure)  = [Bond(at1, at2, bondlength(at1,at2)) for at1 in structure.atoms, at2 in structure.atoms]
bondlength(bond::Bond) = length(bond)

shiftedbonds(r, structure) = [Bond(at1, at2, bondlength(at1, at2, structure.cell' * r)) for at1 in structure.atoms, at2 in structure.atoms]
function shiftedbonds!(bonds, r, structure)
    for i=1:length(bonds)
        pos1 = position(bonds[i].at1)
        pos2 = position(bonds[i].at2)
        bonds[i].length = bondlength(at1, at2, structure.cell' * r)
    end
end
import Base: ==

@inline ==(b1::Bond, b2::Bond) = (norm(length(b1) - length(b2)) < 1e-10 && elsym(b1.at1) == elsym(b2.at1) && elsym(b1.at2) == elsym(b2.at2))

function bondlength(r, o1::Int, o2::Int, structure)
    atoms = structure.atoms
    at1 = orbital2atom(o1, atoms)
    at2 = orbital2atom(o2, atoms)
    return bondlength(at1, at2, structure.cell' * r)
end
