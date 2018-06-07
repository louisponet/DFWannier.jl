import DFControl: searchdir, parse_block, AbstractStructure

struct TbBlock{T<:AbstractFloat}
    Rcart::Vec3{T}
    Rtpiba::Vec3{Int}
    block::Matrix{Complex{T}}
end

struct DipBlock{T<:AbstractFloat}
    Rcart::Vec3{T}
    Rtpiba::Vec3{Int}
    block::Matrix{Point3{T}}
end

mutable struct WanStructure{T<:AbstractFloat} <: AbstractStructure{T}
    name   ::AbstractString
    cell   ::Mat3{T}
    atoms  ::Vector{<:AbstractAtom{T}}
    data   ::Dict{Symbol, Any}
    tbhami ::Vector{TbBlock{T}}
    tbdip  ::Vector{DipBlock{T}}
end

WanStructure(structure::AbstractStructure, tbhami, tbdip) =
    WanStructure(structure.name, structure.cell, structure.atoms, structure.data, tbhami, tbdip)
WanStructure(structure::AbstractStructure,atoms, tbhami, tbdip) =
    WanStructure(structure.name, structure.cell, atoms, structure.data, tbhami, tbdip)
WanStructure(structure::AbstractStructure, tbhami) =
    WanStructure(structure.name, structure.cell, structure.atoms, structure.data, tbhami, Tuple{Int,Int,Int,Int,Int,Complex{T}}[])

structure(str::WanStructure) = Structure(name, cell, atoms, data)
#TODO does not handle the SOC case. Or the case where there is up and down
#TODO handle so that the previous job doesn't get destroyed I mean it's not necessary
#     it also doesn't agree with the paradigm of julia
function add_wan_data(structure::AbstractStructure{T}, job_dir::String) where T
    searchdir(str) = job_dir .* DFControl.searchdir(job_dir, str)
    xsf_files  = searchdir(".xsf")
    hami_file  = searchdir("_hr.dat")[1]
    r_file     = searchdir("_r.dat")[1]

    new_atoms = WanAtom{T}[]
    for at in structure.atoms
        push!(new_atoms, WanAtom(at, WanData(Wfc3D{T}[])))
    end
    t_wfcs = Array{Array{WfcPoint3{T},3},1}(length(xsf_files))
    Threads.@threads for i=1:length(xsf_files)
        t_wfcs[i] = read_xsf_file(xsf_files[i], T)
    end

    for (i, wfc) in enumerate(t_wfcs)
        t_at = new_atoms[1]
        c = calc_dip(wfc, wfc)
        for at in new_atoms[2:end]
            if norm(position(at) - c) < norm(position(t_at) - c)
                t_at = at
            end
        end
        push!(wfcs(t_at), wfc)
    end

    tbhami          = readhami(hami_file, structure)
    tbdip           = read_dipole_file(r_file, structure)
    return WanStructure(structure,new_atoms, tbhami, tbdip)
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
        for str_at in structure.atoms
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
    for at in structure.atoms
        for wfc in wfcs(at)
            push!(out, wfc)
        end
    end
    return out
end

function getwandim(structure::WanStructure{T}) where T
    dim = 0
    for at in structure.atoms
        dim += length(wfcs(at))
    end
    return dim
end
