using DFControl: searchdir, Band, DFBand
import Base: getindex, zero, show, -, +, ==, !=, *, /
# Cleanup Do we really need <:abstractfloat, check this!

"Point of a wavefunction in 3D, holds the complex value of the wavefunction and the cartesian coordinate."
struct WfcPoint3{T<:AbstractFloat}
    w::Complex{T}
    p::Point3{T}
end
+(a::WfcPoint3,b::Point3) = WfcPoint3(a.w,a.p+b)
+(a::WfcPoint3,b::WfcPoint3) = a.p == b.p ? WfcPoint3(a.w+b.w,a.p) : error("Can only sum two wavepoints at the same point in space!")
-(a::WfcPoint3,b::Point3) = WfcPoint3(a.w,a.p-b)
-(a::WfcPoint3,b::WfcPoint3) = a.p == b.p ? WfcPoint3(a.w-b.w,a.p) : error("Can only minus two wavepoints at the same point in space!")
+(a::WfcPoint3{T},b::Complex{T}) where T = WfcPoint3(a.w+b,a.p)
*(a::WfcPoint3,b::AbstractFloat) = WfcPoint3(a.w*b,a.p)
*(a::WfcPoint3{T},b::Complex{T}) where T = WfcPoint3(a.w*b,a.p)
*(a::WfcPoint3{T},b::WfcPoint3{T}) where T = a.p == b.p ? WfcPoint3(a.w*b.w,a.p) : error("Can only times two wavepoints at the same point in space!")
*(b::AbstractFloat,a::WfcPoint3) = WfcPoint3(a.w*b,a.p)
*(b::Complex{T},a::WfcPoint3{T}) where T = WfcPoint3(a.w*b,a.p)
/(a::WfcPoint3{T},b::Complex{T}) where T = WfcPoint3(a.w/b,a.p)
show(io::IO,x::WfcPoint3)=print(io,"w = $(x.w), x = $(x.p[1]), y = $(x.p[2]), z = $(x.p[3])")
zero(::Type{WfcPoint3{T}}) where T<:AbstractFloat = WfcPoint3(zero(Complex{T}),Point3(zero(T)))
zero(x::WfcPoint3{T}) where T<:AbstractFloat = WfcPoint3(zero(Complex{T}), x.p)

const AbstractWfc3D{T} = AbstractArray{WfcPoint3{T}, 3}
Base.zeros(x::AbstractWfc3D) = zero.(x)

function Base.sum(points::AbstractWfc3D{T}) where T
    s = zero(Complex{T})
    for w in points
        s += w.w
    end
    return s
end
function LinearAlgebra.norm(points::AbstractWfc3D{T}) where T
    s = zero(Complex{T})
    for w in points
        s += w.w^2
    end
    return s
end

LinearAlgebra.normalize(points::AbstractWfc3D) = points ./= sqrt(norm(points))
density(wfc::AbstractWfc3D) = wfc .* wfc



"Holds all the calculated values from a wannier model."
mutable struct WannierBand{T<:AbstractFloat} <: Band
    eigvals  ::Vector{T}
    eigvec   ::Vector{Vector{Complex{T}}}
    cms      ::Vector{Point3{T}}
    angmoms  ::Vector{Vector{Point3{T}}}
    spins    ::Vector{Vector{Point3{T}}}
    k_points ::Vector{Vec3{T}}
end

function WannierBand(kpoints::Vector{Vec3{T}}) where T
    klen = length(kpoints)
    WannierBand{T}(zeros(T, klen), fill([zero(Complex{T})], klen), zeros(Point3{T}, klen), fill([zero(Point3{T})], klen), fill([zero(Point3{T})], klen), kpoints)
end

wannierbands(n::Int, kpoints::Vector{<:Vec3}) = [WannierBand(kpoints) for i=1:n]
wannierbands(tbhamis, dfbands::Vector{<:DFBand}) = wannierbands(tbhamis, dfbands)

function wannierbands(tbhamis, kpoints::Vector{<:Vec3})
    matdim = size(tbhamis[1].block)[1]
    outbands = wannierbands(matdim, kpoints)

    for (i, k) in enumerate(kpoints)
        hami = Hk(tbhamis, k)
        eigvals, eigvecs = sorted_eig(hami)
        eigvals_k = real(eigvals)
        for e=1:length(eigvals_k)
            outbands[e].eigvals[i] = eigvals_k[e]
            outbands[e].eigvec[i] = eigvecs[:,e]
            outbands[e].k_points[i] = k
        end
    end
    return outbands
end
wannierbands(tbhamis, dfbands::Vector{<:DFBand}) = wannierbands(tbhamis, dfbands[1].k_points_cryst)



#
# if gpu_enabled
#     mutable struct Wfc3D_gpu{T} <: Wfc{T}
#         grid::CuArray{Tuple{T,T,T},3}
#         values::CuArray{Complex{T},3}
#         # cell::CuArray{Tuple{T,T,T},1}
#         cell::Array{Point3{T},1}
#         atom::Atom{T}
#     end
# end
