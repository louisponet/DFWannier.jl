using DFControl: form_directory,search_dir
import Base: norm, getindex, zero, show, -, +, ==, !=, *, /
# Cleanup Do we really need <:abstractfloat, check this!
"Point of a wavefunction in 3D, holds the complex value of the wavefunction and the cartesian coordinate."
struct WfcPoint3D{T<:AbstractFloat}
    w::Complex{T}
    p::Point3D{T}
end
+(a::WfcPoint3D,b::Point3D) = WfcPoint3D(a.w,a.p+b)
+(a::WfcPoint3D,b::WfcPoint3D) = a.p == b.p ? WfcPoint3D(a.w+b.w,a.p) : error("Can only sum two wavepoints at the same point in space!")
-(a::WfcPoint3D,b::Point3D) = WfcPoint3D(a.w,a.p-b)
+(a::WfcPoint3D{T},b::Complex{T}) where T = WfcPoint3D(a.w+b,a.p)
*(a::WfcPoint3D,b::AbstractFloat) = WfcPoint3D(a.w*b,a.p)
*(a::WfcPoint3D{T},b::Complex{T}) where T = WfcPoint3D(a.w*b,a.p)
*(b::AbstractFloat,a::WfcPoint3D) = WfcPoint3D(a.w*b,a.p)
*(b::Complex{T},a::WfcPoint3D{T}) where T = WfcPoint3D(a.w*b,a.p)
/(a::WfcPoint3D{T},b::Complex{T}) where T = WfcPoint3D(a.w/b,a.p)
show(io::IO,x::WfcPoint3D)=print(io,"w = $(x.w), x = $(x.p[1]), y = $(x.p[2]), z = $(x.p[3])")
zero(::Type{WfcPoint3D{T}}) where T<:AbstractFloat = WfcPoint3D(zero(Complex{T}),Point3D(zero(T)))

# abstract type Wfc{T<:AbstractFloat} end
# "Wavefunction in 3D, holds an array of WfcPoint3D, the superlattice unit cell and the atom around which it lives."
# mutable struct Wfc3D{T} <: Wfc{T}
#     points::Array{WfcPoint3D{T},3}
#     cell::Array{Point3D{T},1}
#     atom::Atom{T}
# end
# /(a::Wfc3D{T},b::Complex{T}) where T = Wfc3D(a.points./b,a.cell,a.atom)
# +(a::Wfc3D,b::Wfc3D) = Wfc3D(a.points+b.points,a.cell,a.atom)
# *(a::Wfc3D,b::AbstractFloat) = Wfc3D(a.points*b,a.cell,a.atom)
# *(a::Wfc3D,b::Complex{AbstractFloat}) = Wfc3D(a.points*b,a.cell,a.atom)
# *(b::AbstractFloat,a::Wfc3D) = Wfc3D(a.points*b,a.cell,a.atom)
# *(b::Complex{T},a::Wfc3D{T}) where T = Wfc3D(a.points*b,a.cell,a.atom)
#
# function norm(a::Wfc3D{T}) where T
#     n = zero(T)
#     for point in a.points
#         n += norm(point.w)^2
#     end
#     return sqrt(n)
# end
#
# function Base.normalize(wfc::Wfc3D{T}) where T
#     n1 = zero(Complex{T})
#     for point in wfc.points
#         n1 += norm(point.w)^2
#     end
#     return wfc/sqrt(n1)
# end
#
# function getindex(x::Wfc3D,i1::Int,i2::Int,i3::Int)
#     return x.points[i1,i2,i3]
# end
# function getindex(x::Wfc3D,i::CartesianIndex{3})
#     return x.points[i[1],i[2],i[3]]
# end
#
# function Base.size(x::Wfc3D)
#     return size(x.points)
# end
# function Base.size(x::Wfc3D,i::Int)
#     return size(x.points,i)
# end
# show(io::IO,x::Wfc3D)=print(io,"Wavefunction Mesh of size($(size(x.points)),\n Physatom = $(x.atom)")

"Holds all the calculated values from a wannier model."
mutable struct WannierBand{T<:AbstractFloat} <: Band
    eigvals  ::Vector{T}
    eigvec   ::Vector{Vector{Complex{T}}}
    cms      ::Vector{Point3D{T}}
    angmoms  ::Vector{Vector{Point3D{T}}}
    spins    ::Vector{Vector{Point3D{T}}}
    k_points ::Vector{Vector{T}}
end

function WannierBand(kpoints::Vector{Vector{T}}) where T
    klen = length(kpoints)
    WannierBand{T}(Vector{T}(klen), Vector{Vector{Complex{T}}}(klen), Vector{Point3D{T}}(klen), Vector{Vector{Point3D{T}}}(klen), Vector{Vector{Point3D{T}}}(klen), kpoints)
end
#
# if gpu_enabled
#     mutable struct Wfc3D_gpu{T} <: Wfc{T}
#         grid::CuArray{Tuple{T,T,T},3}
#         values::CuArray{Complex{T},3}
#         # cell::CuArray{Tuple{T,T,T},1}
#         cell::Array{Point3D{T},1}
#         atom::Atom{T}
#     end
# end

# "Start of any Wannier calculation. Gets constructed by reading the Wannier Hamiltonian and wavefunctions, and gets used in Wannier calculations."
# mutable struct WannierModel{T<:AbstractFloat}
#     hami_raw::Array{Tuple{Int,Int,Int,Int,Int,Complex{T}},1}
#     dip_raw::Array{Tuple{Int,Int,Int,Int,Int,Point3D{T}},1}
#     wfcs::Array{<:Wfc{T},1}
#     k_points::Array{Array{T,1},1}
#     bands::Array{WannierBand{T},1}
#     atoms::Array{Atom{T},1}
#     function WannierModel{T}(dir::String, k_points, atoms::Array{Atom{T},1}) where T<:AbstractFloat
#         dir = form_directory(dir)
#         wfc_files = search_dir(dir, ".xsf")
#         hami_file = search_dir(dir, "_hr.dat")[1]
#         dip_file  = search_dir(dir, "_r.dat")[1]
#         wfcs = Array{Wfc3D{T},1}(length(wfc_files))
#         Threads.@threads for i=1:length(wfcs)
#             wfcs[i] = read_xsf_file(dir*wfc_files[i],atoms[i])
#         end
#         hami_raw = read_hami_file(dir*hami_file,T)
#         dip_raw  = read_dipole_file(dir*dip_file,T)
#         if gpu_enabled
#             wfcs = wfcs .|> host2gpu
#         end
#         return new(hami_raw,dip_raw,wfcs,k_points,WannierBand{T}[],atoms)
#     end
# end
# function WannierModel{T}(dir::String, k_point_file::String, atoms::Array{Atom{T},1},args...) where T<:AbstractFloat
#     k_points = read_ks_from_qe_bands_file(k_point_file,T)[2]
#     return WannierModel{T}(dir,k_points,atoms,args...)
# end
