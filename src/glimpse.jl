using .Glimpse
const Gl = Glimpse
using ColorSchemes
using .Glimpse: RGBf0

struct WfInterface <: Gl.System end

function Gl.update(::WfInterface, m::AbstractLedger)
    if isempty(m[WfManager])
        return
    end
    wfman = Gl.singleton(m, WfManager)
    curvis = wfman.entities[wfman.current]

    gui_func = () -> begin
        Gl.Gui.SetNextWindowPos((650, 200), Gl.Gui.ImGuiCond_FirstUseEver)
        Gl.Gui.SetNextWindowSize((550, 680), Gl.Gui.ImGuiCond_FirstUseEver)
        Gl.Gui.Begin("Wannier Function Manager")
        Gl.Gui.SetWindowFontScale(3.3f0)
        curid = Int32(wfman.current)
        Gl.Gui.@c Gl.Gui.InputInt("Id", &curid)
        if 0 < curid <= length(wfman.entities)
            wfman.current = Int(curid)
        elseif curid <= 0
            wfman.current = length(wfman.entities)
        else
            wfman.current = 1
        end
        Gl.Gui.@c Gl.Gui.InputDouble("dt", &wfman.dt, 0.01, 0.01, "%.3f")
        if wfman.dt <= 0.0
            wfman.dt += 0.01
        end
        if !wfman.iterating
            if Gl.Gui.Button("Iterate")
                wfman.iterating = true
            end
        else
            if Gl.Gui.Button("Stop Iterating")
                wfman.iterating = false
            end
        end
        Gl.Gui.End()
    end
    push!(Gl.singleton(m, Gl.GuiFuncs).funcs, gui_func)

    if wfman.iterating
        wfman.current_t += m[Gl.TimingData][1].dtime
        if wfman.current_t >= wfman.dt
            wfman.current += 1
            if wfman.current >= length(wfman.entities)
                wfman.current = 1
            end
            wfman.current_t = 0.0
        end
    end

    vis = m[Gl.Visible]
    for e in wfman.entities
        vis[e] = Gl.Visible(e == curvis)
    end
end

@component Base.@kwdef mutable struct WfManager
    entities::Vector{Entity}
    wfuncs::Vector{<:WannierFunction}
    current::Int = 1
    dt::Float64 = 0.5
    current_t::Float64 = 0.0
    iterating::Bool = false
end

"""
    visualize_wfuncs(wfuncs::Vector{<:WannierFunction}, str::Structure;
                     iso_ratio = 1/4,
                     alpha     = 0.6,
                     material  = Gl.Material(),
                     phase_channel = Up())

Visualize the wannierfunctions in a Diorama.
The isosurface is determined from the charge, where the coloring signifies the phase of the wavefunction at that point in space.
The `iso_ratio` will be used to determine the isosurface values as the ratio w.r.t the maximum value.
The `phase_channel` denotes whether the phase of the spin up or down channel should be shown in the case of spinor Wannierfunctions. 
"""
function visualize_wfuncs(wfuncs::Vector{<:WannierFunction}, str::Structure; kwargs...)
    dio = Diorama(; background = RGBAf0(60 / 255, 60 / 255, 60 / 255, 1.0f0))
    visualize_wfuncs!(dio, wfuncs, str; kwargs...)
    return dio
end

function visualize_wfuncs!(dio::Diorama, wfuncs, str::Structure;
                           iso_ratio = 1 / 4,
                           alpha = 0.6,
                           material = Gl.Material(),
                           phase_channel = Up())
    DFControl.Display.add_structure!(dio, str)

    phase_id = length(wfuncs[1].values[1]) > 1 ? (phase_channel == Up() ? 1 : 2) : 1
    colfunc(x) = RGBf0(get(ColorSchemes.rainbow, (angle(x[phase_id]) .+ π) / 2π))

    wfentities = fill(Entity(0), length(wfuncs))
    grid = Gl.Grid([Point3f0(w...) for w in wfuncs[1].points])

    for (i, w) in enumerate(wfuncs)
        d = Float32.(norm.(w.values))
        geom = Gl.DensityGeometry(d, iso_ratio * maximum(d))
        color = Gl.DensityColor(colfunc.(w.values))

        wfentities[i] = Entity(dio, Gl.Spatial(), grid, geom, color, material,
                               Gl.Alpha(alpha))
    end

    man = Entity(dio, WfManager(; entities = wfentities, wfuncs = wfuncs))
    insert!(dio, 4, Stage(:wannier, [WfInterface()]))
    update(WfInterface(), dio)
    return dio
end
