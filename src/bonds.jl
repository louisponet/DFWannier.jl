mutable struct Bond{T<:AbstractFloat}
    at1::DFControl.Atom{T}
    at2::DFControl.Atom{T}
    length::T
end

bonds(structure) = [Bond(at1, at2, norm(at1.position - at2.position)) for at1 in structure.atoms, at2 in structure.atoms]


shiftedbonds(r, structure) = [Bond(at1, at2, norm(at1.position - (at2.position + structure.cell' * r))) for at1 in structure.atoms, at2 in structure.atoms]
function shiftedbonds!(bonds, r, structure)
    for i=1:length(bonds)
        pos1 = bonds[i].at1.position
        pos2 = bonds[i].at2.position
        bonds[i].length = norm(pos1 - (pos2 + structure.cell' * r))
    end
end


import Base: ==
@inline ==(b1::Bond, b2::Bond) = (b1.length ≈ b2.length && b1.at1.element.symbol == b2.at1.element.symbol && b1.at2.element.symbol == b2.at2.element.symbol)

bondlength(bond) = bond.length
function bondlength(r, o1::Int, o2::Int, structure)
    atoms = structure.atoms
    at1 = DFControl.orbital2atom(o1, atoms)
    at2 = DFControl.orbital2atom(o2, atoms)
    return bondlength(r, at1, at2, structure.cell)
end
bondlength(r, at1::DFControl.Atom, at2::DFControl.Atom, cell) = norm(at1.position - (cell' * r + at2.position))
