import DFControl: AbstractAtom, Atom, Element, Projection, element, position, elsym, id, pseudo, projections, setpseudo!, atom

mutable struct WanAtData{T <: AbstractFloat}
    lsoc      ::T
    wfcs      ::Vector{Array{WfcPoint3{T}, 3}}
    magmoment ::Vec3{T}
    angmom    ::AbstractMatrix{Vec3{Complex{T}}}
end

WanAtData(wfcs::Vector{Array{WfcPoint3{T}, 3}}) where T = WanAtData(zero(T), wfcs, zero(Vec3{T}), zeros(Vec3{Complex{T}}, 1, 1))

struct WanAtom{T<:AbstractFloat} <: AbstractAtom{T}
    atom    ::Atom{T}
    wandata ::WanAtData{T}
end

WanAtom(atom::Atom{T}, lsoc::T, wfcs::Vector{Array{WfcPoint3{T}, 3}}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, WanAtData(lsoc, wfcs, magmoment, zeros(Vec3{Complex{T}}, 1, 1)))
WanAtom(atom::Atom{T}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, WanAtData(zero(T), Array{WfcPoint3{T}, 3}[], magmoment, zeros(Vec3{Complex{T}}, 1, 1)))
WanAtom(atom::Atom{T}) where T<:AbstractFloat =
    WanAtom(atom, WanAtData(zero(T), Array{WfcPoint3{T}, 3}[], zero(Vec3{T}), zeros(Vec3{Complex{T}}, 1, 1)))
WanAtom(atom_::WanAtom, args...) =
    WanAtom(atom(atom_), args...)

#implementation of the AbstractAtom interface
atom(at::WanAtom) = at.atom

lsoc(atom::WanAtom)      = atom.wandata.lsoc
wfcs(atom::WanAtom)      = atom.wandata.wfcs
magmoment(atom::WanAtom) = atom.wandata.magmoment
angmom(atom::WanAtom)    = atom.wandata.angmom

setlsoc!(atom::WanAtom, lsoc) = (atom.wandata.lsoc = lsoc)
setwfcs!(atom::WanAtom, wfcs) = (atom.wandata.wfcs = wfcs)
setmagmoment!(atom::WanAtom, magmoment) = (atom.wandata.magmoment = magmoment)
setangmom!(atom::WanAtom, angmom) = (atom.wandata.angmom = angmom)

clearangmom!(atom::WanAtom) = setangmom!(atom, zero(angmom(atom)))
