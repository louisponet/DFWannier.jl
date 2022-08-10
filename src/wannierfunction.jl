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

for f in (:size, :getindex, :setindex!)
    @eval @inline @propagate_inbounds Base.$f(x::WannierFunction, i...) =
        Base.$f(x.values, i...)
end

for f in (:length, :stride, :ndims, :axes, :strides)
    @eval @inline Base.$f(w::WannierFunction) = Base.$f(w.values)
end

Base.similar(x::WannierFunction,::Type{S}) where S = 
  WannierFunction(x.points, similar(x.values, S))

Base.unsafe_convert(T::Type{<:Ptr}, x::WannierFunction) =
    unsafe_convert(T, x.values)


Base.Broadcast.broadcastable(w::WannierFunction) =
    w.values

#### LinearAlgebra overloads
function LinearAlgebra.adjoint(w::WannierFunction)
    out = WannierFunction(w.points, similar(w.values))
    adjoint!(out, w)
end

LinearAlgebra.adjoint!(w1::WannierFunction, w2::WannierFunction) =
    w1 .= adjoint.(w2)

function LinearAlgebra.dot(w1::WannierFunction{T}, w2::WannierFunction{T}) where {T}
    s = zero(T)
    for (v1, v2) in zip(w1.values, w2.values)
        s += v1' * v2
    end
    return real(s)
end

function LinearAlgebra.dot(v::Vector, wfs::Vector{<:WannierFunction})
    res = WannierFunction(wfs[1].points, zeros(eltype(wfs[1].values), size(wfs[1].values)))
    for ic in 1:length(v)
        res .+= v[ic] .* wfs[ic]
    end
    return res
end
LinearAlgebra.dot(wfs::Vector{<:WannierFunction}, v::Vector) =
    dot(v, wfs)

LinearAlgebra.norm(wfc::WannierFunction) =
    dot(wfc, wfc)

LinearAlgebra.normalize!(wfc::WannierFunction) =
    wfc ./= sqrt(norm(wfc))

same_grid(w1::WannierFunction, w2::WannierFunction) =
    w1.points === w2.points

function wan_op(op::Function, w1::W, w2::W) where {W <: WannierFunction}
    @assert same_grid(w1, w2) "Wannier functions are not defined on the same grid"
    return WannierFunction(w1.points, op(w1.values, w2.values))
end

Base.:(+)(w1::WannierFunction, w2::WannierFunction) = wan_op(+, w1, w2)
Base.:(*)(w1::WannierFunction, w2::WannierFunction) = wan_op(*, w1, w2)
Base.:(-)(w1::WannierFunction, w2::WannierFunction) = wan_op(-, w1, w2)
Base.:(*)(w1::WannierFunction, n::Number) = WannierFunction(w1.points, w1.values .* n)
Base.:(*)(n::Number, w1::WannierFunction) = WannierFunction(w1.points, n .* w1.values)
Base.:(/)(w1::WannierFunction, n::Number) = WannierFunction(w1.points, n ./ w1.values)
Base.:(/)(n::Number, w1::WannierFunction) = WannierFunction(w1.points, w1.values ./ n)

LinearAlgebra.dot(w1::WannierFunction, n::Number) = w1 * n
LinearAlgebra.dot(n::Number, w1::WannierFunction) = n * w1

function generate_wannierfunctions(k_filenames::Vector{String}, chk_info, wannier_plot_supercell::NTuple{3,Int}, wan_plot_list=1:chk_info.n_wann)
    num_kpts = length(chk_info.kpoints)
    U = permutedims(chk_info.U_matrix, (2, 1, 3))
    U_opt = permutedims(chk_info.U_matrix_opt,(2,1,3))
    tu = read_unk(k_filenames[1])
    nrx, nry, nrz = size(tu,1), size(tu,2), size(tu,3)
    supercell_xrange = -div(wannier_plot_supercell[1],2)*nrx : div(wannier_plot_supercell[1] + 1, 2)*nrx - 1
    supercell_yrange = -div(wannier_plot_supercell[2],2)*nry : div(wannier_plot_supercell[2] + 1, 2)*nry - 1
    supercell_zrange = -div(wannier_plot_supercell[3],2)*nrz : div(wannier_plot_supercell[3] + 1, 2)*nrz - 1

    nx, ny, nz = length.((supercell_xrange, supercell_yrange, supercell_zrange))

    nwfun = length(wan_plot_list)
    wfuncs_all = zeros(eltype(tu), nwfun, (wannier_plot_supercell .* (nrx, nry, nrz))...,size(tu, 5))
    n_wann = chk_info.n_wann
    r_wan      = zeros(eltype(tu), chk_info.n_wann, nrx, nry, nrz)

    p = Progress(length(chk_info.kpoints))
    @inbounds for ik = 1:length(chk_info.kpoints)
        k = chk_info.kpoints[ik]
        unk_all = read_unk(k_filenames[ik])
        for is = 1:size(tu, 5)
            u = U[wan_plot_list, :, ik]
            u_opt = U_opt[:, :, ik]
            inc_ids = findall(!iszero, chk_info.lwindow[:, ik])
            fill!(r_wan, 0.0)
            for ib in 1:chk_info.ndimwin[ik]
                iib = inc_ids[ib]
                Threads.@threads for nz in 1:nrz
                    for ny in 1:nry
                        for nx in 1:nrx
                            @simd for iw in 1:n_wann
                                r_wan[iw, nx, ny, nz] += u_opt[iw, ib] * unk_all[nx,ny,nz,iib,is]
                            end
                        end
                    end
                end
            end
            Threads.@threads for iisz in 1:nz
                isz = supercell_zrange[iisz]
                iz = mod1(isz, nrz)
                for iisy in 1:ny
                    isy = supercell_yrange[iisy]
                    iy = mod1(isy, nry)
                    for iisx in 1:nx
                        isx = supercell_xrange[iisx]
                        ix = mod1(isx, nrx)
                        scalfac = exp(2im*π*dot(k, Vec3((isx-1)/nrx, (isy-1)/nry, (isz-1)/nrz)))
                        for ib in 1:n_wann
                            rt = r_wan[ib, ix, iy, iz] * scalfac
                            for iw in 1:nwfun
                                wfuncs_all[iw, iisx, iisy, iisz, is] += u[iw, ib] * rt
                            end
                        end
                    end
                end
            end
        end
        next!(p)
    end
    wfuncs_all ./= num_kpts
    if size(tu, 5) == 1
        Threads.@threads for iw = 1:size(wfuncs_all, 1)
            tmaxx = 0.0
            cmod  = 1.0+0.0im
            for iisx in 1:nx
                for iisy in 1:ny
                    for iisz in 1:nz
                        w = wfuncs_all[iw, iisx, iisy, iisz, 1]
                        t = abs2(w)
                        if t > tmaxx
                            tmaxx = t
                            cmod = w
                        end
                    end
                end
            end
            cmod /= abs(cmod)
            @views wfuncs_all[:, :, :, iw, 1] ./= cmod
        end
    end
    str_cell = ustrip.(chk_info.cell)
    points = [str_cell * Point3((x-1)/nrx, (y-1)/nry, (z-1)/nrz) for x in supercell_xrange, y in supercell_yrange, z in supercell_zrange]
    if size(tu,5) == 1
        wfuncs_out = Vector{WannierFunction{1, eltype(wfuncs_all).parameters[1]}}(undef, size(wfuncs_all, 1))
        Threads.@threads for i=1:size(wfuncs_all, 1)
            wfuncs_out[i] = WannierFunction{1, eltype(wfuncs_all).parameters[1]}(points, map(x -> SVector(x), view(wfuncs_all,i, :, :, :, 1)))
        end
        return normalize!.(wfuncs_out)
    else
        wfuncs_out = Vector{WannierFunction{2, eltype(wfuncs_all).parameters[1]}}(undef, size(wfuncs_all, 1))
        Threads.@threads for i=1:size(wfuncs_all, 1)
            wfuncs_out[i] = WannierFunction{2, eltype(wfuncs_all).parameters[1]}(points, map(x -> SVector(x), zip(view(wfuncs_all, i, :, :, :, 1), view(wfuncs_all, i, :, :, :, 2))))
        end

        return wfuncs_out
    end
end

function generate_wannierfunctions(job::Job, supercell::NTuple{3,Int}, args...)
    tdir = job.dir
    unk_files = reverse(searchdir(job, "UNK"))
    chk_files = reverse(searchdir(job, ".chk"))
    if !DFC.Jobs.runslocal(job)
        tdir = mkpath(tempname())
        for f in [unk_files; chk_files]
            fname = splitpath(f)[end]
            DFC.Servers.pull(job, fname, joinpath(tdir, fname))
        end
        unk_files = reverse(searchdir(tdir, "UNK"))
        chk_files = reverse(searchdir(tdir, ".chk"))
    end
    if ismagnetic(job.structure) && Structures.iscolin(job.structure) && !any(Calculations.issoc, job.calculations)
        wfuncs = Vector{WannierFunction}[]
        for (is, s) in enumerate(("up", "down"))
            wan_calc  = getfirst(x -> eltype(x) == Wannier90&& x[:spin] == s, job.calculations)
            chk_info  = read_chk(joinpath(tdir, "$(wan_calc.name).chk"))
            unk_files = filter(x->occursin(".$is", x), searchdir(tdir, "UNK"))
            push!(wfuncs, generate_wannierfunctions(unk_files, chk_info, supercell, args...))
        end
        wfuncs = (up=wfuncs[1], down=wfuncs[2])
    else
        wan_calc  = getfirst(x -> eltype(x)==Wannier90, job.calculations)
        chk_info  = read_chk(joinpath(tdir, "$(wan_calc.name).chk"))
        unk_files = searchdir(tdir, "UNK")
        wfuncs = generate_wannierfunctions(unk_files, chk_info, supercell, args...)
    end
    if !DFC.Jobs.runslocal(job)
        rm(tdir, recursive=true)
    end
    return wfuncs
end

function bloch_sum(wfunc, kpoint; i_pos_offset = (0,0,0), i_neg_offset=(0,0,0))
    cell_boundaries = div.(size(wfunc.points), 3) .+ 1
    x = wfunc.points[cell_boundaries[1]+1, 1, 1] .- wfunc.points[1]
    y = wfunc.points[1, cell_boundaries[2]+1, 1] .- wfunc.points[1]
    z = wfunc.points[1, 1, cell_boundaries[3]+1] .- wfunc.points[1]
    bloch = WannierFunction(wfunc.points, zeros(eltype(wfunc.values), size(wfunc.values)))
    dims = size(wfunc.values)
    for i1 in -3:1:3, i2 in -3:1:3, i3 in -3:1:3
        R_cryst = Vec3(i1, i2, i3)
        o1, o2, o3 = cell_boundaries .* R_cryst
        shiftvec = x * R_cryst[1] .+ y * R_cryst[2] .+ z*R_cryst[3]
        phase = ℯ^(2im*π*(R_cryst ⋅ kpoint)) 
        if i1 + i2 + i3 == 0
            continue
        end
        if i1 < 0
            o1 += i_neg_offset[1]
        elseif i1 > 0
            o1 += i_pos_offset[1]
        end
        if i2 < 0
            o2 += i_neg_offset[2]
        elseif i2 > 0
            o2 += i_pos_offset[2]
        end
        if i3 < 0
            o3 += i_neg_offset[3]
        elseif i3 > 0
            o3 += i_pos_offset[3]
        end
            
        for j3 in 1:dims[3]
            oid3 = mod1(j3 - o3, dims[3])
            for j2 in 1:dims[2]
                oid2 = mod1(j2 - o2, dims[2])
                for j1 in 1:dims[1]
                    oid1 = mod1(j1 - o1, dims[1])
                   
                    bloch.values[j1, j2, j3] += phase * wfunc.values[oid1, oid2, oid3]
                end
            end
        end
    end
    return bloch
end

"Calculates the angular momentum between two wavefunctions and around the center."
function calc_angmom(wfc1::WannierFunction{N, T}, wfc2::WannierFunction{N, T}, center::Point3{T}, cutoff=Inf) where {N, T <: AbstractFloat}
    points = wfc1.points
    origin = points[1, 1, 1]
    da     = points[2, 1, 1] - origin
    db     = points[1, 2, 1] - origin
    dc     = points[1, 1, 2] - origin
    V      = SMatrix{3,3}(inv([convert(Array, da) convert(Array, db) convert(Array, dc)])')
    L      = zero(Point3{Complex{T}})
    c2 = cutoff^2
    @inbounds for i2 in 2:size(wfc1, 3)
        for i1 in 2:size(wfc1, 2)
            for i in 2:size(wfc1, 1)
                r       = points[i, i1, i2] - center
                if dot(r, r) < c2
                    dw_cryst = Point3(wfc2.values[i, i1, i2] - wfc2.values[i-1, i1,   i2],
                                      wfc2.values[i, i1, i2] - wfc2.values[i,   i1-1, i2],
                                              wfc2.values[i, i1, i2] - wfc2.values[i,   i1,   i2-1])

                    dw_cart = V * dw_cryst
                    L      += (wfc1.values[i, i1, i2]',) .* cross(r, dw_cart)
                end
            end
        end
    end
    return -1im * L
end

#this doesn't work I think
# function calc_angmom_squared(wfc1::WannierFunction{N, T}, wfc2::WannierFunction{N, T}, center::Point3{T}) where {N, T <: AbstractFloat}
#        points = wfc1.points
#     origin = points[1, 1, 1]
#     da     = points[2, 1, 1] - origin
#     db     = points[1, 2, 1] - origin
#     dc     = points[1, 1, 2] - origin
#     V      = SMatrix{3,3}(inv([convert(Array, da) convert(Array, db) convert(Array, dc)])')
#     Lsq      = zero(Complex{T})

#     @inbounds for i2 = 2:size(wfc1)[3]
#         for i1 = 2:size(wfc1)[2]
#             for i = 2:size(wfc1)[1]
#                 dw_cryst = Point3(wfc2.values[i, i1, i2] - wfc2.values[i-1, i1,   i2],
#                                   wfc2.values[i, i1, i2] - wfc2.values[i,   i1-1, i2],
#                                                          wfc2.values[i, i1, i2] - wfc2.values[i,   i1,   i2-1])
#                    dw_cryst_sq = map(x->x .^2,dw_cryst)

#                 r       = points[i, i1, i2] - center
#                 dw_cart = V * dw_cryst
#                 Lsq    += wfc1.values[i, i1, i2] ⋅ (r[2]^2 * (dw_cryst_sq[1] + dw_cryst_sq[3]) +
#                                                     r[1]^2 * (dw_cryst_sq[2] + dw_cryst_sq[3]) +
#                                                     r[3]^2 * (dw_cryst_sq[1] + dw_cryst_sq[2]) -
#                                                     2 * (r ⋅ dw_cryst +
#                                                          r[2] * r[3] * dw_cryst[2] .* dw_cryst[3] +
#                                                          r[1] * r[3] * dw_cryst[1] .* dw_cryst[3] +
#                                                          r[1] * r[2] * dw_cryst[1] .* dw_cryst[2])) 



#                end
#            end
#        end
#        return Lsq
# end

function calc_spin(wfc1::WannierFunction{2, T}, wfc2::WannierFunction{2, T}) where T <: AbstractFloat
    S = Point3(SMatrix{2, 2}(0, 1, 1, 0)/2, 
               SMatrix{2, 2}(0, -1im, 1im, 0)/2,
               SMatrix{2, 2}(1, 0, 0, -1)/2)

    outS = zero(Point3{Complex{T}})
    for (w1, w2) in zip(wfc1.values, wfc2.values)
        outS += (w1',) .* S .* (w2,)
    end
    return outS
end


"Calculates the dipole term between two wavefunctions. Make sure the wavefunctions are normalized!"
function calc_dip(wfc1::WannierFunction{N, T}, wfc2::WannierFunction{N, T}) where {N, T <: AbstractFloat}
    out = zero(Point3{Complex{T}})
    for (w1, w2, p) in zip(wfc1, wfc2, wfc1.points)
        out += w1' * w2 * p
    end
    return real(out)
end

# Is this code actually correct?
# "Calculates the dipoles from the supplied wannier dipole output."
# function calc_k_dips(dip_raw::Array{Tuple{Int, Int, Int, Int, Int, Point3{T}}}, kpoints::AbstractArray) where T<:AbstractFloat
#        dim = 0
#        for i=1:length(dip_raw)
#               d = dip_raw[i][4]
#               if d > dim
#                      dim = d
#               else
#                      break
#               end
#        end
#        out = zeros(Point3{T}, dim, dim)
#        tmp = [[zeros(Complex{T}, 3)] for i=1:dim, i1=1:dim]
#        for i=1:size(dip_raw)[1]
#               d = dip_raw[i]
#               complex_part = 2π*(kpoints[1]*d[1]+kpoints[2]*d[2]+kpoints[3]*d[3])
#               factor = exp(-1im * complex_part)
#               tmp[d[4],d[5]][1] += d[6][1] * factor
#               tmp[d[4],d[5]][2] += d[6][2] * factor
#               tmp[d[4],d[5]][3] += d[6][3] * factor
#        end
#        for i in eachindex(out)
#               out[i] = Point3(real(tmp[i][1]),real(tmp[i][2]),real(tmp[i][3]))
#        end
#        return Mat{2*dim, 2*dim, Point3{T}}([out zeros(out);zeros(out) out])
# end
