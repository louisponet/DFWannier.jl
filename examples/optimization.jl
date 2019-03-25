using DFWannier
using Optim
using DataStructures
function optimize_l()
    T = Float64

    job = ldj("/home/ponet/HfO2/NSOC/")
    wanjob = add_wan_data(job)

    k_points = qe_read_ks_from_bands_file("/home/ponet/HfO2/SOC/bands.out")[2]
    test_eig, , , = calc_observables(wanjob.structure, kpoints[1])

    bands = qe_read_bands_file("/home/ponet/HfO2/SOC/bands.out", T)[b:e]

    at_ids = getfield.(unique(wanjob.structure.atoms), :id)

    lsocs = OrderedDict{Symbol, T}()
    for id in at_ids
        lsocs[id] = zero(T)
    end
    function f(lambdas)
        for (key, l) in zip(keys(lsocs), lambdas)
            lsocs[key] = l
        end
        set_soc!(wanjob.structure, lsocs...)

        energy, , , = calc_observables(wanjob.structure, k_points)

        out_diff = 0.0
        for i=1:length(bands)
            calcbands = [val[i] for val in energy]
            for (calc, exact) in zip(calcbands[80:120], bands[i].eigvals[80:120])
                out_diff += abs(calc-exact)
            end
        end
        return out_diff
    end
    return optimize(f, [lambdaHf, lambdaO], show_trace = true, show_every = 1, extended_trace=true)
end
results = optimize_l()
println(results)
