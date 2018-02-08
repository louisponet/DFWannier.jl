import DFControl: Element, Projection

mutable struct WanAtom{T<:AbstractFloat} <: AbstractAtom{T}
    id          ::Symbol
    element     ::Element
    position    ::Point3D{T}
    pseudo      ::String
    projections ::Array{Projection, 1}
    lsoc        ::T
    wfcs        ::Vector{Array{WfcPoint3D{T}, 3}}
    magmoment   ::SVector{3, T}
end
WanAtom(atom::Atom{T}, lsoc, wfcs, magmoment) where T<:AbstractFloat =
    WanAtom{T}(atom.id, atom.element, atom.position, atom.pseudo, atom.projections, lsoc, wfcs, magmoment)
WanAtom(atom::Atom{T}, magmoment) where T =
    WanAtom(atom, zero(T), Array{WfcPoint3D{T}, 3}[], magmoment)
WanAtom(atom::Atom{T}, lsoc, wfcs) where T =
    WanAtom(atom, lsoc, wfcs, SVector(zeros(T,3)...))
WanAtom(atom::Atom{T}) where T =
    WanAtom(atom, zero(T), Array{WfcPoint3D{T}, 3}[], SVector(zeros(T,3)...))
