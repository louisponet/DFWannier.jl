function optimspins(angles::Vector{T}, rs, exchanges, H, anisotropy) where T
    nexch = length(exchanges)
    spins = Vec{3, T}[]
    nat   = length(rs)
    for (r, θ, ϕ) in zip(rs, angles[1:nat], angles[nat+1:end])
        push!(spins, polar2xyz(r, θ, ϕ))
    end
    return heisenberg_energy(spins, exchanges, H, anisotropy)
end

function polarization(spins, g=2.0)
    polarization = -g*( dot(spins[12]-spins[11], spins[5]  + spins[7])
                       +dot(spins[19]-spins[9] , spins[6]  + spins[10]))
                       # +dot(spins[15]-spins[16], spins[17] + spins[18])
                       # +dot(spins[8] -spins[20], spins[13] + spins[14]))

    return polarization
end

function calcspinspols(atoms::Vector{WanAtom{T}}, exchanges, Hs, aniz, startconfigs=4, itps=20, noisestrength=0.00001,startE=0.0) where T
    nstrength = T(noisestrength)
    rs      = [at.magmoment[3] for at in atoms]
    f(x, h) = optimspins(x, rs, exchanges, h, aniz)
    nexch   = length(exchanges)
    nat     = length(atoms)
    angs0   = [randθsϕs(T, nat) for i =1:startconfigs]

    totangs    = Vector{Vector{T}}[]
    totconfigs = Vector{Vector{Vec{3, T}}}[]
    totpols    = Vector{T}[]
    energies   = T[]
    min = T(startE)
    for (hi, h) in enumerate(Hs)
        prevangs = isempty(totangs) ? angs0 : totangs[end]
        td = TwiceDifferentiable(x -> f(x, h), angs0[1], autodiff=:forward)

        tangs    = Vector{T}[]
        tconfigs = Vector{Vec{3, T}}[]
        tpols    = T[]
        max = hi==1? 1 : itps
        for i=1:max
            noise = randθsϕs(T, nat) .* nstrength
            for (ia, angles) in enumerate(prevangs)
                length(tconfigs) == 4 && break
                # a0    = angles
                a0    = angles + noise

                res   = optimize(td, a0, Newton())
                energy= Optim.minimum(res)
                if energy < min || energy ≈ min
                    min = energy
                    as  = Optim.minimizer(res)
                    testspin = polar2xyz(atoms[1].magmoment[3], as[1], as[nat+1])
                    test= isempty(tconfigs) || all([norm(config[1] - testspin) > 0.1 for config in tconfigs])
                    if test
                        println("H=$h, it=$i, iprevang=$ia, energy=$energy")
                        spins = Vec{3, T}[]
                        for (r, θ, ϕ) in zip(rs, as[1:nat], as[nat+1:end])
                            push!(spins, polar2xyz(r, θ, ϕ))
                        end
                        push!(tconfigs, spins)
                        push!(tpols, polarization(spins))
                        push!(tangs, as)
                    end
                end
            end
        end
        push!(totangs, tangs)
        push!(totconfigs, tconfigs)
        push!(totpols, tpols)
        push!(energies, min)
    end
    return totconfigs, totpols, energies
end
