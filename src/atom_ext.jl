import DFControl: Element, Projection

mutable struct WanAtom{T<:AbstractFloat} <: AbstractAtom{T}
    id          ::Symbol
    element     ::Element
    position    ::Point3{T}
    pseudo      ::String
    projections ::Array{Projection, 1}
    lsoc        ::T
    wfcs        ::Vector{Array{WfcPoint3{T}, 3}}
    magmoment   ::Vec3{T}
    angmom      ::AbstractMatrix{Vec3{Complex{T}}}
    function WanAtom(atom::Atom{T}, args::Pair...) where T <: AbstractFloat
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
                    if typeof(value) == typ
                        setfield!(out, field, value)
                    else
                        setfield!(out, field, convert(typ, value))
                    end
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
WanAtom(atom::Atom{T}, lsoc::T, wfcs::Vector{Array{WfcPoint3{T}, 3}}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, :lsoc => lsoc, :wfcs => wfcs, :magmoment => magmoment)
WanAtom(atom::Atom{T}, magmoment::Vec3{T}) where T<:AbstractFloat =
    WanAtom(atom, zero(T), Array{WfcPoint3{T}, 3}[], magmoment)
WanAtom(atom::Atom{T}, lsoc, wfcs) where T =
    WanAtom(atom, :lsoc => lsoc, :wfcs => wfcs, :magmoment => SVector(zeros(T,3)...))
