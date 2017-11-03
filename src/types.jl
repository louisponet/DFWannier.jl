# Cleanup Do we really need <:abstractfloat, check this!
"Point in 3D space in cartesian coordinates with specified float type"
struct Point3D{T<:AbstractFloat}<:AbstractFloat
  x::T
  y::T
  z::T
end
Point3D(::Type{T},x) where T<:AbstractFloat = Point3D(T(x),T(x),T(x))
"Point of a wavefunction in 3D, holds the complex value of the wavefunction and the cartesian coordinate."
struct WfcPoint3D{T<:AbstractFloat}
  w::Complex{T}
  p::Point3D{T}
end

"Atom in 3D space, has a center in cartesian coordinates and a parameter for the spin-orbit coupling strength."
struct PhysAtom{T<:AbstractFloat}
  center::Point3D{T}
  l_soc::T
end
PhysAtom(x,y,z,l_soc)                                  = PhysAtom(Point3D(x,y,z),l_soc)
PhysAtom(::Type{T},x,y,z,l_soc) where T<:AbstractFloat = PhysAtom(T(x),T(y),T(z),T(l_soc))
PhysAtom(::Type{T}) where T<:AbstractFloat             = PhysAtom(Point3D(T,0.0),T(0.0))

"Wavefunction in 3D, holds an array of WfcPoint3D, the superlattice unit cell and the atom around which it lives."
mutable struct Wfc3D{T<:AbstractFloat}
  points::Array{WfPoint3D{T},3}
  cell::Array{Point3D{T},1}
  atom::PhysAtom{T}
end