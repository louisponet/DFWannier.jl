import DFControl: AbstractAtom, Element, Projection, element, position, elsym, id, pseudo, projections, setpseudo!

mutable struct WanAtData{T <: AbstractFloat}
    lsoc      ::T
    wfcs      ::Vector{Array{WfcPoint3{T}, 3}}
    magmoment ::Vec3{T}
    angmom    ::AbstractMatrix{Vec3{Complex{T}}}
end

WanData(wfcs::Vector{Array{WfcPoint3{T}, 3}}) where T = WanData(zero(T), wfcs, zero(Vec3{T}), zeros(Vec3{Complex{T}}, 1,1))

struct WanAtom{T<:AbstractFloat} <: AbstractAtom{T}
    atom    ::Atom{T}
    wandata ::WanAtData{T}
end

WanAtom(atom::Atom{T}, lsoc::T, wfcs::Vector{Array{WfcPoint3{T}, 3}}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, WanAtData(lsoc, wfcs, magmoment, zeros(Vec3{Complex{T}}, 1, 1))
WanAtom(atom::Atom{T}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, WanAtData(zero(T), Array{WfcPoint3{T}, 3}[], magmoment,zeros(Vec3{Complex{T}}, 1,1) ))

#implementation of the AbstractAtom interface
position(atom::WanAtom)    = position(atom.atom)
element(atom::WanAtom)     = element(atom.atom)
id(atom::WanAtom)          = id(atom.atom)
pseudo(atom::WanAtom)      = pseudo(atom.atom)
projections(atom::WanAtom) = projections(atom.atom)
setpseudo!(atom::WanAtom, pseudo)  = setpseudo!(atom.atom, pseudo)
setprojections!(atom::WanAtom, projections)  = setprojections!(atom.atom, projections)

lsoc(atom::WanAtom)      = atom.wandata.lsoc
wfcs(atom::WanAtom)      = atom.wandata.wfcs
magmoment(atom::WanAtom) = atom.wandata.magmoment
angmom(atom::WanAtom)    = atom.wandata.angmom


function setlsoc!(atom::WanAtom, lsoc) atom.wandata.lsoc = lsoc end
function setwfcs!(atom::WanAtom, wfcs) atom.wandata.wfcs = wfcs end
function setmagmoment!(atom::WanAtom, magmoment) atom.wandata.magmoment = magmoment end
function setangmom!(atom::WanAtom, setangmom) atom.wandata.angmom = angmom end
