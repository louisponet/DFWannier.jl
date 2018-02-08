import DFControl: search_dir, parse_block

function add_wan_data!(job::DFJob, T=Float64)
    add_wan_data!(job.structure, job.local_dir, T)
    return job
end

#TODO does not handle the SOC case. Or the case where there is up and down
function add_wan_data!(structure::Structure, job_dir::String, T=Float64)
    search_dir(str) = job_dir .* DFControl.search_dir(job_dir, str)
    xsf_files  = search_dir(".xsf")
    hami_file  = search_dir("hr.dat")[1]
    r_file     = search_dir("r.dat")[1]
    wout_file  = search_dir(".wout")[1]

    t_wfcs = Array{Array{WfcPoint3D{T},3},1}(length(xsf_files))
    Threads.@threads for i=1:length(xsf_files)
        t_wfcs[i] = read_xsf_file(xsf_files[i], T)
    end

    centers= nothing
    open(wout_file, "r") do f
        readuntil(f, "Final State")

        centers = Point3D{T}.(parse_block(f, T, T, T))
    end

    for at in structure.atoms
        at.wfcs = Wfc3D{T}[]
    end
    for (i, (c, wfc)) in enumerate(zip(center, wfcs))
        t_at = structure.atoms[1]
        for at in structure.atoms[2:end]
            if norm(at.position - c) < norm(t_at.position - c)
                t_at = at
            end
        end
        push!(t_at.wfcs, wfc)
    end


    structure.data[:tbhami] = read_hami_file.(hami_files, T)
    structure.data[:tbdip]  = read_dipole_file.(r_files, T)
    return structure
end
