using DFControl: Band, Point3, Vec3, Point, Mat3
using DFControl.Utils: searchdir
using DFControl: Atom, Element, Projection, element, position 
import DFControl: Length
# Cleanup Do we really need <:abstractfloat, check this!
 

struct OperatorBlock{T <: AbstractFloat}
	L::Vector{Matrix{Complex{T}}}
	S::Vector{Matrix{Complex{T}}}
    J::Vector{Matrix{Complex{T}}}
end

getindex(A::Matrix, a1::T , a2::T) where {T<:Union{Atom, Projection}} =
	getindex(A, range(a1), range(a2))

getindex(A::Matrix, a::Atom) =
	getindex(A, a, a)

getindex(A::Vector, a::Atom) =
	getindex(A, range(a))

view(A::Matrix, a1::T, a2::T) where {T<:Union{Atom, Projection}} =
	view(A, range(a1), range(a2))

view(A::Matrix, a::Union{Atom, Projection}) =
	view(A, range(a), range(a))

view(A::Vector, a::Union{Atom, Projection}) =
	view(A, range(a))

struct RmnBlock{T<:AbstractFloat}
    R_cart  ::Vec3{T}
    R_cryst ::Vec3{Int}
    block   ::Matrix{Point3{T}}
end

const TbRmn{T} = Vector{RmnBlock{T}}

struct SiteDiagonalD{T<:AbstractFloat}
	values ::Vector{T}
	T      ::Matrix{Complex{T}}
end

# view(D::SiteDiagonalD, at::Atom) =
# 	()


# mutable struct WanStructure{T<:AbstractFloat, LT<:Length{T}} <: AbstractStructure{T, LT}
#     structure ::Structure{T, LT}
#     tbhamis   ::Vector{TbHami{T}}
#     tbRmns    ::Vector{TbRmn{T}}
# end

# WanStructure(structure::Structure, wan_atoms::Vector{<:WanAtom}, tbhamis, tbrmns) =
#     WanStructure(Structure(structure, wan_atoms), tbhamis, tbrmns)

# WanStructure(structure::Structure{T}, tbhamis) where T =
#     WanStructure(structure, tbhamis, TbRmn{T}[])
# WanStructure(structure_::WanStructure, args...) =
#     WanStructure(structure(structure_), args...)

# structure(str::WanStructure) =
	# str.structure
# #TODO does not handle the SOC case. Or the case where there is up and down
# #TODO handle so that the previous job doesn't get destroyed I mean it's not necessary
# #     it also doesn't agree with the paradigm of julia
# function add_wan_data(structure::AbstractStructure{T}, job_dir::String, threaded=true) where T
#     searchdir(str) =
	# job_dir .* DFControl.searchdir(job_dir, str)
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

# function add_wan_data(job::Job)
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
#     setsoc!(job::Job, socs...)

# Accepts a varargs list of atom symbols => soc,
# which will set the soc of the atoms in the job structure to the specified values.
# """
# function setsoc!(job::Job, socs...)
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

struct ThreadCache{T}
	caches::Vector{T}
	ThreadCache(orig::T) where {T} =
		new{T}([deepcopy(orig) for i = 1:nthreads()])
end

@inline cache(t::ThreadCache) =
	t.caches[threadid()]

for f in (:getindex, :setindex!, :copyto!, :size, :length, :iterate, :sum, :view, :fill!)
	@eval Base.$f(t::ThreadCache{<:AbstractArray}, i...) = Base.$f(cache(t), i...)
end

for op in (:+, :-, :*, :/)
	@eval $op(t::ThreadCache{T}, v::T) where {T} = $op(cache(t), v)
	@eval $op(v::T, t::ThreadCache{T}) where {T} = $op(v, cache(t))
end

fillall!(t::ThreadCache{<:AbstractArray{T}}, v::T)   where {T} =
	fill!.(t.caches, (v,))

gather(t::ThreadCache) = sum(t.caches)

LinearAlgebra.mul!(t1::ThreadCache{T}, v::T, t2::ThreadCache{T}) where {T<:AbstractArray} =
	mul!(cache(t1), v, cache(t2))

LinearAlgebra.mul!(t1::T, v::T, t2::ThreadCache{T}) where {T<:AbstractArray} =
	mul!(t1, v, cache(t2))

LinearAlgebra.mul!(t1::ThreadCache{T}, t2::ThreadCache{T}, t3::ThreadCache{T}) where {T<:AbstractArray} =
	mul!(cache(t1), cache(t2), cache(t3))

LinearAlgebra.adjoint!(t1::ThreadCache{T}, v::T) where {T} =
	adjoint!(cache(t1), v)

####
#### Broadcasting
####
Base.ndims(::Type{ThreadCache{T}}) where {T<:AbstractArray} =
	ndims(T)
 # Base.broadcast(f, As::ThreadCache...) =
	# broadcast(f, getindex.(getfield.(As, :caches), threadid()))
Base.Broadcast.broadcastable(tc::ThreadCache{<:AbstractArray}) =
	cache(tc)
# Base.Broadcast.BroadcastStyle(::Type{ThreadCache{T}}) where {T<:AbstractArray} =
	# Base.Broadcast.BroadcastStyle(T)




