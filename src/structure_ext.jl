import DFControl: searchdir, parse_block, AbstractStructure, getfirst, structure, Structure, read_wannier_output
struct TbBlock{T<:AbstractFloat}
    Rcart::Vec3{T}
    Rtpiba::Vec3{Int}
    block::Matrix{Complex{T}}
end
const TbHami{T} = Vector{TbBlock{T}}

struct RmnBlock{T<:AbstractFloat}
    Rcart::Vec3{T}
    Rtpiba::Vec3{Int}
    block::Matrix{Point3{T}}
end
const TbRmn{T} = Vector{RmnBlock{T}}

mutable struct WanStructure{T<:AbstractFloat} <: AbstractStructure{T}
    structure ::Structure{T}
    tbhamis   ::Vector{TbHami{T}}
    tbRmns    ::Vector{TbRmn{T}}
end

WanStructure(structure::Structure, wan_atoms::Vector{<:WanAtom}, tbhamis, tbrmns) =
    WanStructure(Structure(structure, wan_atoms), tbhamis, tbrmns)

WanStructure(structure::Structure{T}, tbhamis) where T =
    WanStructure(structure, tbhamis, TbRmn{T}[])
WanStructure(structure_::WanStructure, args...) =
    WanStructure(structure(structure_), args...)

structure(str::WanStructure) = str.structure
#TODO does not handle the SOC case. Or the case where there is up and down
#TODO handle so that the previous job doesn't get destroyed I mean it's not necessary
#     it also doesn't agree with the paradigm of julia
function add_wan_data(structure::AbstractStructure{T}, job_dir::String, threaded=true) where T
    searchdir(str) = job_dir .* DFControl.searchdir(job_dir, str)
    xsf_files   = searchdir(".xsf")
    hami_files  = reverse(searchdir("_hr.dat")) #such that dn is last
    r_files     = reverse(searchdir("_r.dat")) #such that dn is last
    centers     = [c[:center] for c in read_wannier_output(searchdir(".wout")[1])[:final_state]]
    new_atoms = WanAtom{T}[]
    for at in atoms(structure)
        push!(new_atoms, WanAtom(at, WanAtData(Wfc3D{T}[])))
    end
    t_wfcs = Vector{Array{WfcPoint3{T},3}}(undef, length(xsf_files))
    if threaded
        Threads.@threads for i=1:length(xsf_files)
            t_wfcs[i] = read_xsf_file(T, xsf_files[i])
        end
    else
        for i=1:length(xsf_files)
            t_wfcs[i] = read_xsf_file(T, xsf_files[i])
        end
    end

    for (i, (wfc, c)) in enumerate(zip(t_wfcs, centers))
        t_at = new_atoms[1]
        for at in new_atoms[2:end]
            if norm(position(at) - c) < norm(position(t_at) - c)
                t_at = at
            end
        end
        push!(wfcs(t_at), wfc)
    end
    tbhamis = Vector{TbHami{T}}(undef, length(hami_files))
    tbrmns  = Vector{TbRmn{T}}(undef, length(r_files))
    Threads.@threads for i=1:length(hami_files)
        tbhamis[i] = readhami(hami_files[i], structure)
    end
    Threads.@threads for i=1:length(r_files)
        tbrmns[i] = read_rmn_file(r_files[i], structure)
    end
    return WanStructure(structure, new_atoms, tbhamis, tbrmns)
end

function add_wan_data(job::DFJob)
    job.structure = add_wan_data(job.structure, job.local_dir)
    return job
end

"""
    setsoc!(structure::Structure, socs...)

Accepts a varargs list of atom symbols => soc,
which will set the soc of the atoms in the structure to the specified values.
"""
function setsoc!(structure::AbstractStructure{T}, socs...) where T
    for (at, soc) in socs
        for str_at in atoms(structure)
            if id(str_at) == at
                setlsoc!(str_at, T(soc))
            end
        end
    end
    return structure
end

"""
    setsoc!(job::DFJob, socs...)

Accepts a varargs list of atom symbols => soc,
which will set the soc of the atoms in the job structure to the specified values.
"""
function setsoc!(job::DFJob, socs...)
    setsoc!(job.structure, socs...)
    return job
end

"""
    wfcs(structure::WanStructure)

Returns the wavefunctions that are linked to the atoms inside the structure.
"""
function wfcs(structure::WanStructure{T}) where T
    out = Array{WfcPoint3{T}, 3}[]
    for at in atoms(structure)
        for wfc in wfcs(at)
            push!(out, wfc)
        end
    end
    return out
end

function getwandim(structure::WanStructure{T}) where T
    dim = 0
    for at in atoms(structure)
        dim += length(wfcs(at))
    end
    return dim
end
