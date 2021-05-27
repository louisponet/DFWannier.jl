using DFControl: searchdir, Band, DFBand, Point3, Vec3, Point, Mat3
using DFControl: AbstractAtom, Atom, Element, Projection, element, position, elsym, pseudo, projections, set_pseudo!, atom
import DFControl: Length
import Base: getindex, zero, show, -, +, ==, !=, *, /, view
# Cleanup Do we really need <:abstractfloat, check this!

struct WannierFunction{N, T<:AbstractFloat} <: AbstractArray{SVector{N, Complex{T}}, 3}
	points::Array{Point{3, T}, 3}
	values::Array{SVector{N, Complex{T}}, 3}
end

function WannierFunction(filename::AbstractString, points::Array{Point3{T}, 3}=read_points_from_xsf(filename)) where {T <: AbstractFloat}
	re = read_values_from_xsf(T, filename)
	values = [SVector(complex(a)) for a in re]
	return normalize(WannierFunction(points, values))
end

function WannierFunction(filename_re::String, filename_im::String, points::Array{Point3{T}, 3} = read_points_from_xsf(filename_re)) where {T <: AbstractFloat}
	re, im = read_values_from_xsf.(T, (filename_re, filename_im))
	values = [SVector(Complex(a, b)) for (a, b) in zip(re, im)]
	return normalize(WannierFunction(points, values))
end

function WannierFunction(filename_up_re::String, filename_up_im::String, filename_down_re::String, filename_down_im::String, points::Array{Point3{T}, 3} = read_points_from_xsf(filename_up_re)) where {T <: AbstractFloat}
	up_re, up_im, down_re, down_im =
		read_values_from_xsf.(T, (filename_up_re, filename_up_im, filename_down_re, filename_down_im))
	values = [SVector(Complex(a, b), Complex(c, d)) for (a, b, c, d) in zip(up_re, up_im, down_re, down_im)]
	return normalize(WannierFunction(points, values))
end

WannierFunction(point_func::Function,  points::Array) =
    normalize(WannierFunction(points, point_func.(points)))


values(w::WannierFunction) =
	w.values

for f in (:size, :getindex, :setindex!)
	@eval @inline @propagate_inbounds Base.$f(x::WannierFunction, i...) =
		Base.$f(values(x), i...)
end

for f in (:length, :stride, :ndims, :axes, :strides)
	@eval @inline Base.$f(w::WannierFunction) = Base.$f(values(w))
end

Base.similar(x::WannierFunction,::Type{S}) where S = 
  WannierFunction(x.points, similar(values(x), S))

Base.unsafe_convert(T::Type{<:Ptr}, x::WannierFunction) =
	unsafe_convert(T, values(x))


Base.Broadcast.broadcastable(w::WannierFunction) =
	values(w)

#### LinearAlgebra overloads
function LinearAlgebra.adjoint(w::WannierFunction)
	out = WannierFunction(w.points, similar(values(w)))
	adjoint!(out, w)
end

LinearAlgebra.adjoint!(w1::WannierFunction, w2::WannierFunction) =
	w1 .= adjoint.(w2)

function LinearAlgebra.dot(w1::WannierFunction{T}, w2::WannierFunction{T}) where {T}
    s = zero(T)
    for (v1, v2) in zip(values(w1), values(w2))
        s += v1' * v2
    end
    return real(s)
end

LinearAlgebra.norm(wfc::WannierFunction) =
	dot(wfc, wfc)

LinearAlgebra.normalize!(wfc::WannierFunction) =
	wfc ./= sqrt(norm(wfc))

####
same_grid(w1::WannierFunction, w2::WannierFunction) =
	w1.points === w2.points 

function wan_op(op::Function, w1::W, w2::W) where {W <: WannierFunction}
	@assert same_grid(w1, w2) "Wannier functions are not defined on the same grid"
	return WannierFunction(w1.points, op(w1.values, w2.values))
end

+(w1::WannierFunction, w2::WannierFunction) = wan_op(+, w1, w2)
*(w1::WannierFunction, w2::WannierFunction) = wan_op(*, w1, w2)
-(w1::WannierFunction, w2::WannierFunction) = wan_op(-, w1, w2)

*(w1::WannierFunction, n::Number) = WannierFunction(w1.points, w1.values .* n)
*(n::Number, w1::WannierFunction) = WannierFunction(w1.points, n .* w1.values)
/(w1::WannierFunction, n::Number) = WannierFunction(w1.points, n ./ w1.values)
/(n::Number, w1::WannierFunction) = WannierFunction(w1.points, w1.values ./ n)

struct OperatorBlock{T <: AbstractFloat}
	L::Vector{Matrix{Complex{T}}}
	S::Vector{Matrix{Complex{T}}}
    J::Vector{Matrix{Complex{T}}}
end

struct WanAtom{T <: AbstractFloat, LT <: Length{T}} <: AbstractAtom{T, LT}
    atom    ::Atom{T, LT}
    wandata ::Dict{Symbol, <:Any}
end

getindex(at::WanAtom, s::Symbol) =
	getindex(at.wandata, s::Symbol)

# #implementation of the AbstractAtom interface
DFControl.atom(at::WanAtom) =
	at.atom

getindex(A::Matrix, a1::T , a2::T) where {T<:Union{AbstractAtom, Projection}} =
	getindex(A, range(a1), range(a2))

getindex(A::Matrix, a::AbstractAtom) =
	getindex(A, a, a)

getindex(A::Vector, a::AbstractAtom) =
	getindex(A, range(a))

view(A::Matrix, a1::T, a2::T) where {T<:Union{AbstractAtom, Projection}} =
	view(A, range(a1), range(a2))

view(A::Matrix, a::Union{AbstractAtom, Projection}) =
	view(A, range(a), range(a))

view(A::Vector, a::Union{AbstractAtom, Projection}) =
	view(A, range(a))

import DFControl: searchdir, parse_block, AbstractStructure, getfirst, structure, Structure, wan_read_input
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

# view(D::SiteDiagonalD, at::AbstractAtom) =
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




