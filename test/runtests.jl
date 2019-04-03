using DFWannier
using Base.Test

# px =normalize([WfcPoint3(Complex(p[1]*ℯ^-norm(p)), p) for p in points])
# py =normalize([WfcPoint3(Complex(p[2]*ℯ^-norm(p)), p) for p in points])
# pz =normalize([WfcPoint3(Complex(p[3]*ℯ^-norm(p)), p) for p in points])

# points = [DFW.Point3(a, b, c) for a=-10:0.1:10, b=-10:0.1:10, c=-10:0.1:10]

# px = normalize(DFW.WannierFunction(points, [SVector(Complex(p[1] * ℯ^-norm(p)), zero(Complex{Float64})) for p in points]))
# py = normalize(DFW.WannierFunction(points, [SVector(zero(Complex{Float64}), Complex(p[2] * ℯ^-norm(p))) for p in points]))
# pz = normalize(DFW.WannierFunction(points, [SVector(Complex(p[3] * ℯ^-norm(p)), zero(Complex{Float64})) for p in points]))
# pz = normalize(DFW.WannierFunction(points, [SVector(Complex(p[3] * ℯ^-norm(p)), zero(Complex{Float64})) for p in points]))
# pzdn = normalize(DFW.WannierFunction(points, [SVector(zero(Complex{Float64}), Complex(p[3] * ℯ^-norm(p))) for p in points]))

# test = normalize(px+1im*py)

# @time calc_angmom(test, test, zero(DFW.Point3{Float64}))
# DFW.calc_spin(pz, pzdn)
# write your own tests here
commutator(A1::Matrix{T}, A2::Matrix{T}) where T = A1*A2 - A2*A1

