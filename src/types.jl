using DFControl: searchdir, Band, DFBand, Point3, Vec3, Point, Mat3
import DFControl: AbstractAtom, Atom, Element, Projection, element, position, elsym, id, pseudo, projections, setpseudo!, atom
import Base: getindex, zero, show, -, +, ==, !=, *, /
# Cleanup Do we really need <:abstractfloat, check this!

struct WannierFunction{N, T<:AbstractFloat}
	points::Array{Point{3, T}, 3}
	values::Array{SVector{N, Complex{T}}, 3}
end
Base.size(w::WannierFunction) = size(w.points)

function WannierFunction(filename_re::String, filename_im::String, points::Array{Point3{T}, 3}) where {T <: AbstractFloat}
	values = [SVector(Complex(a, b)) for (a, b) in zip(read_values_from_xsf(T, filename_re), read_values_from_xsf(T, filename_im))]
	return normalize(WannierFunction(points, values))
end

function WannierFunction(filename_up_re::String, filename_up_im::String, filename_down_re::String, filename_down_im::String, points::Array{Point3{T}, 3}) where {T <: AbstractFloat}
	values = [SVector(Complex(a, b), Complex(c, d)) for (a, b, c, d) in zip(read_values_from_xsf(T, filename_up_re),
																	              read_values_from_xsf(T, filename_up_im),
																	              read_values_from_xsf(T, filename_down_re),
																	              read_values_from_xsf(T, filename_down_im))]
	return normalize(WannierFunction(points, values))
end

+(w1::W, w2::W) where {W <: WannierFunction} = W(w1.points, w1.values + w2.values)
-(w1::W, w2::W) where {W <: WannierFunction} = W(w1.points, w1.values - w2.values)
*(w1::W, f::Number) where {W <: WannierFunction} = W(w1.points, f * w1.values)
*(f::Number, w1::W) where {W <: WannierFunction}= W(w1.points, f * w1.values)

function LinearAlgebra.norm(wfc::WannierFunction{T}) where T
    s = zero(T)
    for v in wfc.values
        s += v' * v
    end
    return real(s)
end

LinearAlgebra.normalize(wfc::WannierFunction) = WannierFunction(wfc.points, wfc.values ./= sqrt(norm(wfc)))

# Base.@kwdef struct OperatorBlock{T <: AbstractFloat}
# 	projection ::Projection
# 	L::Vector{Matrix{Complex{T}}} =
# 		[zeros(Complex{T}, orbsize(projection), orbsize(projection))
# 		      for i = 1:3]
# 	S::Vector{Matrix{Complex{T}}} =
# 		[zeros(Complex{T}, orbsize(projection), orbsize(projection))
# 		      for i = 1:3]
#     J::Vector{Matrix{Complex{T}}} = L .+ S


struct OperatorBlock{T <: AbstractFloat}
	L::Vector{Matrix{Complex{T}}}
	S::Vector{Matrix{Complex{T}}}
    J::Vector{Matrix{Complex{T}}}
end

struct WanAtom{T <: AbstractFloat} <: AbstractAtom{T}
    atom    ::Atom{T}
    wandata ::Dict{Symbol, <:Any}
end

Base.getindex(at::WanAtom, s::Symbol) = getindex(at.wandata, s::Symbol)
# WanAtom(atom::Atom{T}, lsoc::T, wfcs::Vector{Array{WfcPoint3{T}, 3}}, magmoment::Vec3{T}) where T<:AbstractFloat =
#     WanAtom(atom, WanAtData(lsoc, wfcs, magmoment, zeros(Vec3{Complex{T}}, 1, 1)))
# WanAtom(atom::Atom{T}, magmoment::Vec3{T}) where T<:AbstractFloat =
#     WanAtom(atom, WanAtData(zero(T), Array{WfcPoint3{T}, 3}[], magmoment, zeros(Vec3{Complex{T}}, 1, 1)))
# WanAtom(atom::Atom{T}) where T<:AbstractFloat =
#     WanAtom(atom, WanAtData(zero(T), Array{WfcPoint3{T}, 3}[], zero(Vec3{T}), zeros(Vec3{Complex{T}}, 1, 1)))
# WanAtom(atom_::WanAtom, args...) =
#     WanAtom(atom(atom_), args...)

# #implementation of the AbstractAtom interface
atom(at::WanAtom) = at.atom

# lsoc(atom::WanAtom)                     = atom.wandata.lsoc
# wfcs(atom::WanAtom)                     = atom.wandata.wfcs
# magmoment(atom::WanAtom)                = atom.wandata.magmoment
# angmom(atom::WanAtom)                   = atom.wandata.angmom
# setlsoc!(atom::WanAtom, lsoc)           = (atom.wandata.lsoc      = lsoc)
# setwfcs!(atom::WanAtom, wfcs)           = (atom.wandata.wfcs      = wfcs)
# setmagmoment!(atom::WanAtom, magmoment) = (atom.wandata.magmoment = magmoment)
# setangmom!(atom::WanAtom, angmom)       = (atom.wandata.angmom    = angmom)
# clearangmom!(atom::WanAtom)             = setangmom!(atom, zero(angmom(atom)))

import DFControl: searchdir, parse_block, AbstractStructure, getfirst, structure, Structure, read_wannier_output
struct TbBlock{T <: AbstractFloat}
    R_cart  ::Vec3{T}
    R_cryst ::Vec3{Int}
    block   ::Matrix{Complex{T}}
end
Base.getindex(h::TbBlock, i)    = getindex(h.block, i)
Base.getindex(h::TbBlock, i, j) = getindex(h.block, i, j)
LinearAlgebra.eigen(h::TbBlock) = eigen(h.block)
Base.size(h::TbBlock)           = size(h.block)

const TbHami{T}                  = Vector{TbBlock{T}}
Base.eltype(::TbHami{T}) where T = T

get_block(h::TbHami, R::Vec3{Int}) = getfirst(x->x.R_cryst == R, h)
blockdim(h::TbHami) = size(h[1].block)
empty_block(h::TbHami{T}) where T = Matrix{Complex{T}}(undef, blockdim(h))

struct RmnBlock{T<:AbstractFloat}
    R_cart  ::Vec3{T}
    R_cryst ::Vec3{Int}
    block   ::Matrix{Point3{T}}
end
const TbRmn{T} = Vector{RmnBlock{T}}

mutable struct WanStructure{T<:AbstractFloat} <: AbstractStructure{T}
    structure ::Structure{T}
    tbhamis   ::Vector{TbHami{T}}
    tbRmns    ::Vector{TbRmn{T}}
end

# WanStructure(structure::Structure, wan_atoms::Vector{<:WanAtom}, tbhamis, tbrmns) =
#     WanStructure(Structure(structure, wan_atoms), tbhamis, tbrmns)

# WanStructure(structure::Structure{T}, tbhamis) where T =
#     WanStructure(structure, tbhamis, TbRmn{T}[])
# WanStructure(structure_::WanStructure, args...) =
#     WanStructure(structure(structure_), args...)

# structure(str::WanStructure) = str.structure
# #TODO does not handle the SOC case. Or the case where there is up and down
# #TODO handle so that the previous job doesn't get destroyed I mean it's not necessary
# #     it also doesn't agree with the paradigm of julia
# function add_wan_data(structure::AbstractStructure{T}, job_dir::String, threaded=true) where T
#     searchdir(str) = job_dir .* DFControl.searchdir(job_dir, str)
#     xsf_files   = searchdir(".xsf")
#     hami_files  = reverse(searchdir("_hr.dat")) #such that dn is last
#     r_files     = reverse(searchdir("_r.dat")) #such that dn is last
#     centers     = [c[:center] for c in read_wannier_output(searchdir(".wout")[1])[:final_state]]
#     new_atoms = WanAtom{T}[]
#     for at in atoms(structure)
#         push!(new_atoms, WanAtom(at, WanAtData(Wfc3D{T}[])))
#     end
#     t_wfcs = Vector{Array{WfcPoint3{T},3}}(undef, length(xsf_files))
#     if threaded
#         Threads.@threads for i=1:length(xsf_files)
#             t_wfcs[i] = read_xsf_file(T, xsf_files[i])
#         end
#     else
#         for i=1:length(xsf_files)
#             t_wfcs[i] = read_xsf_file(T, xsf_files[i])
#         end
#     end

#     for (i, (wfc, c)) in enumerate(zip(t_wfcs, centers))
#         t_at = new_atoms[1]
#         for at in new_atoms[2:end]
#             if norm(position(at) - c) < norm(position(t_at) - c)
#                 t_at = at
#             end
#         end
#         push!(wfcs(t_at), wfc)
#     end
#     tbhamis = Vector{TbHami{T}}(undef, length(hami_files))
#     tbrmns  = Vector{TbRmn{T}}(undef, length(r_files))
#     Threads.@threads for i=1:length(hami_files)
#         tbhamis[i] = readhami(hami_files[i], structure)
#     end
#     Threads.@threads for i=1:length(r_files)
#         tbrmns[i] = read_rmn_file(r_files[i], structure)
#     end
#     return WanStructure(structure, new_atoms, tbhamis, tbrmns)
# end

# function add_wan_data(job::DFJob)
#     job.structure = add_wan_data(job.structure, job.local_dir)
#     return job
# end

# """
#     setsoc!(structure::Structure, socs...)

# Accepts a varargs list of atom symbols => soc,
# which will set the soc of the atoms in the structure to the specified values.
# """
# function setsoc!(structure::AbstractStructure{T}, socs...) where T
#     for (at, soc) in socs
#         for str_at in atoms(structure)
#             if name(str_at) == at
#                 setlsoc!(str_at, T(soc))
#             end
#         end
#     end
#     return structure
# end

# """
#     setsoc!(job::DFJob, socs...)

# Accepts a varargs list of atom symbols => soc,
# which will set the soc of the atoms in the job structure to the specified values.
# """
# function setsoc!(job::DFJob, socs...)
#     setsoc!(job.structure, socs...)
#     return job
# end

# """
#     wfcs(structure::WanStructure)

# Returns the wavefunctions that are linked to the atoms inside the structure.
# """
# function wfcs(structure::WanStructure{T}) where T
#     out = Array{WfcPoint3{T}, 3}[]
#     for at in atoms(structure)
#         for wfc in wfcs(at)
#             push!(out, wfc)
#         end
#     end
#     return out
# end

# function getwandim(structure::WanStructure{T}) where T
#     dim = 0
#     for at in atoms(structure)
#         dim += length(wfcs(at))
#     end
#     return dim
# end

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

import Base: +, -, *, /
struct ThreadCache{T}
	orig::T
	caches::Vector{T}
end

ThreadCache(orig) = ThreadCache(orig, [deepcopy(orig) for i = 1:nthreads()])

Base.getindex(t::ThreadCache{<:AbstractArray}, i...) = t.caches[threadid()][i...]
Base.setindex!(t::ThreadCache{<:AbstractArray{T}}, v::T, i...) where T = t.caches[threadid()][i...] = v

+(t::ThreadCache{T}, v::T) where T = t.caches[threadid()] + v 
-(t::ThreadCache{T}, v::T) where T = t.caches[threadid()] - v 
*(t::ThreadCache{T}, v::T) where T = t.caches[threadid()] * v 
/(t::ThreadCache{T}, v::T) where T = t.caches[threadid()] / v 
+(v::T, t::ThreadCache{T}) where T = v - t.caches[threadid()]  
-(v::T, t::ThreadCache{T}) where T = v * t.caches[threadid()]  
*(v::T, t::ThreadCache{T}) where T = v / t.caches[threadid()]  
/(v::T, t::ThreadCache{T}) where T = v + t.caches[threadid()]

Base.size(t::ThreadCache) = size(t.orig)
Base.length(t::ThreadCache) = length(t.orig)
Base.iterate(t::ThreadCache) = iterate(t.caches[threadid()])
Base.iterate(t::ThreadCache, p) = iterate(t.caches[threadid()], p)
Base.sum(t::ThreadCache) = sum(t.caches)
Base.view(t::ThreadCache, v...) = view(t.caches[threadid()], v...)
Base.fill!(t::ThreadCache{<:AbstractArray{T}}, v::T) where T = fill!(t.caches[threadid()], v)
LinearAlgebra.mul!(t1::ThreadCache{T}, v::T, t2::ThreadCache{T}) where T = mul!(t1.caches[threadid()], v, t2.caches[threadid()]) 
LinearAlgebra.mul!(t1::T, v::T, t2::ThreadCache{T}) where T = mul!(t1, v, t2.caches[threadid()]) 
LinearAlgebra.adjoint!(t1::ThreadCache{T}, v::T) where T = adjoint!(t1.caches[threadid()], v)

Base.broadcast(f, As::ThreadCache...) = broadcast(f, getindex.(getfield.(As, :caches), threadid())) 
