import DFControl: AbstractAtom, Atom, Element, Projection, element, position, elsym, id, pseudo, projections, setpseudo!

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
    WanAtom(atom, WanAtData(lsoc, wfcs, magmoment, zeros(Vec3{Complex{T}}, 1, 1)))
WanAtom(atom::Atom{T}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, WanAtData(zero(T), Array{WfcPoint3{T}, 3}[], magmoment,zeros(Vec3{Complex{T}}, 1,1)))

#implementation of the AbstractAtom interface
atom(at::WanAtom) = at.atom

lsoc(atom::WanAtom)      = atom.wandata.lsoc
wfcs(atom::WanAtom)      = atom.wandata.wfcs
magmoment(atom::WanAtom) = atom.wandata.magmoment
angmom(atom::WanAtom)    = atom.wandata.angmom


function setlsoc!(atom::WanAtom, lsoc) atom.wandata.lsoc = lsoc end
function setwfcs!(atom::WanAtom, wfcs) atom.wandata.wfcs = wfcs end
function setmagmoment!(atom::WanAtom, magmoment) atom.wandata.magmoment = magmoment end
function setangmom!(atom::WanAtom, setangmom) atom.wandata.angmom = angmom end
