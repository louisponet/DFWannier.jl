using DFControl: searchdir, Band, DFBand, Point3, Vec3, Point, Mat3
using DFControl: AbstractAtom, Atom, Element, Projection, element, position, elsym, pseudo, projections, setpseudo!, atom
using DFControl.Unitful: Length
import Base: getindex, zero, show, -, +, ==, !=, *, /, view
# Cleanup Do we really need <:abstractfloat, check this!

struct WannierFunction{N, T<:AbstractFloat} <: AbstractArray{SVector{N, Complex{T}}, 3}
	points::Array{Point{3, T}, 3}
	values::Array{SVector{N, Complex{T}}, 3}
end

function WannierFunction(filename_re::String, filename_im::String, points::Array{Point3{T}, 3}) where {T <: AbstractFloat}
	re, im = read_values_from_xsf.(T, (filename_re, filename_im))
	values = [SVector(Complex(a, b)) for (a, b) in zip(re, im)]
	return normalize(WannierFunction(points, values))
end

function WannierFunction(filename_up_re::String, filename_up_im::String, filename_down_re::String, filename_down_im::String, points::Array{Point3{T}, 3}) where {T <: AbstractFloat}

	up_re, up_im, down_re, down_im =
		read_values_from_xsf.(T, (filename_up_re, filename_up_im, filename_down_re, filename_down_im))

	values = [SVector(Complex(a, b), Complex(c, d)) for (a, b, c, d) in zip(up_re, up_im, down_re, down_im)]
	return normalize(WannierFunction(points, values))
end

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

LinearAlgebra.normalize(wfc::WannierFunction) =
	wfc ./= sqrt(norm(wfc))

####

same_grid(w1::WannierFunction, w2::WannierFunction) =
	w1.points === w2.points 

function wan_op(op::Function, w1::W, w2::W) where {W <: WannierFunction}
	@assert same_grid(w1, w2) "Wannier functions are not defined on the same grid"
	op(w1, w2)
end

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

struct TbBlock{T <: AbstractFloat, M <: AbstractMatrix{Complex{T}}, MI<:AbstractMatrix{Int}, VS <: Vector{Vec3{Int}}, LT<:Length{T}}
    R_cart  ::Vec3{LT}
    R_cryst ::Vec3{Int}
    wigner_seitz_shifts::VS
    # Like w90 irdist_ws: The integer number of unit cells to shift the Wannier function j to put its center inside the wigner-seitz of wannier function i. Can have multiple equivalent shifts (maximum of 8), they are all stored. 
    wigner_seitz_nshifts::MI
    wigner_seitz_degeneracy::Int #not sure if we need to keep this
    # For example on boundaries of the supercell
    block   ::M
end


block(x::TbBlock) = x.block

for f in (:getindex, :size, :similar)
	@eval Base.$f(h::TbBlock, args...) = $f(block(h), args...)
end

LinearAlgebra.eigen(h::TbBlock) =
	eigen(block(h))

const TbHami{T, M, MI, VS, LT}  = Vector{TbBlock{T, M, MI, VS, LT}}

getindex(h::TbHami, R::Vec3{Int}) =
	getfirst(x -> x.R_cryst == R, h)

#some small type piracy?
Base.zeros(m::AbstractArray{T}) where {T} =
	fill!(similar(m), zero(T))

zeros_block(h::TbHami) =
	zeros(block(h[1]))

similar_block(h::TbHami) =
	similar(block(h[1]))

blocksize(h::TbHami, args...) =
	size(block(h[1]), args...)

for op in (:+, :-, :*, :/)
	@eval $op(t::TbBlock{T}, v::T) where {T} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), v))
	@eval $op(v::T, t::TbBlock{T}) where {T} = TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)))
	@eval $op(t::TbBlock{T,M}, v::M) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), v))
	@eval $op(v::M, t::TbBlock{T,M}) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(v, block(t)))
	@eval $op(t::TbBlock{T,M}, v::TbBlock{T,M}) where {T,M} = TbBlock(t.R_cart, t.R_cryst, $op(block(t), block(v)))
end


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


mutable struct WanStructure{T<:AbstractFloat, LT<:Length{T}} <: AbstractStructure{T, LT}
    structure ::Structure{T, LT}
    tbhamis   ::Vector{TbHami{T}}
    tbRmns    ::Vector{TbRmn{T}}
end

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

"Holds all the calculated values from a wannier model."
@with_kw mutable struct WannierBand{T<:AbstractFloat} <: Band
    kpoints_cryst ::Vector{Vec3{T}}
    eigvals        ::Vector{T}
    eigvec         ::Vector{Vector{Complex{T}}}
    cms      ::Vector{Point3{T}} = Point3{T}[]
    angmoms  ::Vector{Vector{Point3{T}}} = Vector{Point3{T}}[]
    spins    ::Vector{Vector{Point3{T}}} = Vector{Point3{T}}[]
end

function WannierBand(kpoints::Vector{Vec3{T}}, dim::Int) where T
    klen = length(kpoints)
    WannierBand{T}(kpoints_cryst=kpoints, eigvals=zeros(T, klen), eigvec=[zeros(Complex{T}, dim) for k=1:klen])
end

wannierbands(kpoints::Vector{<:Vec3}, dim::Int) =
	[WannierBand(kpoints, dim) for i=1:dim]

function wannierbands(tbhamis::TbHami, kpoints::Vector{<:Vec3})
    matdim = blocksize(tbhamis, 2)
    outbands = wannierbands(kpoints, matdim)
	calc_caches = [EigCache(block(tbhamis[1])) for i = 1:nthreads()]
    for i = 1:length(kpoints)
	    tid  = threadid()
	    k    = kpoints[i]
	    c    = calc_caches[tid]
        hami = Hk(tbhamis, k)
        eigvals, eigvecs = eigen(hami, c)
        for e=1:length(eigvals)
            outbands[e].eigvals[i] = eigvals[e]
            outbands[e].eigvec[i] = eigvecs[:,e]
            outbands[e].kpoints_cryst[i] = k
        end
    end
    return outbands
end
wannierbands(tbhamis, dfbands::Vector{<:DFBand}) =
	wannierbands(tbhamis, dfbands[1].k_points_cryst)

function character_contribution(wband::WannierBand, atoms::Vector{<:AbstractAtom})
    contributions = zeros(length(wband.kpoints_cryst))
    for (i, v) in enumerate(wband.eigvec)
        for a in atoms
            contributions[i] += real(sum(conj.(v[a]) .* v[a]))
        end
    end
    return contributions
end

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




