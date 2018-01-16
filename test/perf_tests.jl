using DFWannier
using ProfileView
T = Float32
x = WannierModel{T}("/Users/ponet/Documents/Fysica/PhD/GeTe/colin/paperxsf/test1","/Users/ponet/Documents/Fysica/PhD/GeTe/fullrel/GeTe_bands.out",[[Atom(T[0.0,0.0,-0.0239129,-0.155854]...) for i=1:4]...,[Atom(T[0.0,0.0,5.5540692,0.318205]...) for i=1:4]...]);
@time tbbandssoc = calculate_eig_cm_angmom_soc(x,90:0.2:110);
Profile.clear()
using BenchmarkTools
@benchmark tbbandssoc = calculate_eig_cm_angmom_soc(x,90:0.2:110)
@profile tbbandssoc1 = calculate_eig_cm_angmom_soc(x,50:0.2:110);
@time calculate_angmom(x.wfcs[1],x.wfcs[3])
Profile.clear()
Profile.init(1000000000, 0.00100000)
ProfileView.view()
@code_warntype read_xsf_file("/Users/ponet/Documents/Fysica/PhD/GeTe/colin/paperxsf/test1")


struct perfTest
  val::Float64
  function perfTest()
    new(rand(Float64))
  end
end

struct perfTest2
  val::Float64
  t::Bool
  function perfTest2()
    new(rand(Float64),rand(Bool))
  end
end

const test_array = [perfTest() for i=1:10000000]
const test_array2 = [perfTest2() for i=1:10000000]

function bench_test(t)
  out = 0.0
  @inbounds for i=1:10000000
    out += t[i].val
  end
  out
end
using StaticArrays
function bench_test2(t)
  out = 0.0
  v = Array{perfTest2,1}(12)
  @inbounds for i=1:10:10000000
    unsafe_copy!(v,1,t,i,12)
    @inbounds for j = 1:12
    out += v[j].val
    end
  end
  out
end
function bench_test2(t)
  out = 0.0
  @inbounds for i=1:10000000
    out += t[i].val
  end
  out
end

using BenchmarkTools
@code_native bench_test(test_array)
@code_native bench_test2(test_array2)
