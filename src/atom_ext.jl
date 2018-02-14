import DFControl: Element, Projection

mutable struct WanAtom{T<:AbstractFloat} <: AbstractAtom{T}
    id          ::Symbol
    element     ::Element
    position    ::Point3{T}
    pseudo      ::String
    projections ::Array{Projection, 1}
    lsoc        ::T
    wfcs        ::Vector{Array{WfcPoint3{T}, 3}}
    magmoment   ::SVector{3, T}
    angmom      ::AbstractMatrix{Vec3{Complex{T}}}
    function WanAtom(atom::Atom{T}, args...) where T <: AbstractFloat
        out          = new{T}()
        for f in fieldnames(Atom)
            if isdefined(atom, f) && f in fieldnames(WanAtom)
                setfield!(out, f, getfield(atom, f))
            end
        end
        names = fieldnames(WanAtom)
        types = fieldtype.(WanAtom, names)
        for (name, typ) in zip(names[4:end], types[4:end])
            found = false
            for (field, value) in args
                if field == name
                    setfield!(out, field, convert(typ, value))
                    found = true
                end
            end
            if !found
                try
                    setfield!(out, field, zero(typ))
                end
            end
        end
        return out
    end
end
WanAtom(atom::Atom{T}, lsoc, wfcs, magmoment) where T<:AbstractFloat =
    WanAtom(atom, :lsoc => lsoc, :wfcs => wfcs, :magmoment => magmoment)
WanAtom(atom::Atom{T}, magmoment) where T =
    WanAtom(atom, zero(T), Array{WfcPoint3{T}, 3}[], :magmoment => magmoment)
WanAtom(atom::Atom{T}, lsoc, wfcs) where T =
    WanAtom(atom, :lsoc => lsoc, :wfcs => wfcs, :magmoment => SVector(zeros(T,3)...))
