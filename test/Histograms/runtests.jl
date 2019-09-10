using Test
import NPZ

include("../../src/Histograms.jl")
const Hgm = Histograms

const data_dir = joinpath(@__DIR__, "data")
const x1_bal = NPZ.npzread(joinpath(data_dir, "x1_bal.npy"))
const x1_dns = NPZ.npzread(joinpath(data_dir, "x1_dns.npy"))
const x1_onl = NPZ.npzread(joinpath(data_dir, "x1_onl.npy"))
const w1_dns_bal        = 0.03755967829782972
const w1_dns_onl        = 0.004489688974663949
const w1_bal_onl        = 0.037079734072606625
const w1_dns_bal_unnorm = 0.8190688772401341

################################################################################
# unit testing #################################################################
################################################################################
@testset "unit testing" begin
  arr1 = [1, 1, 1, 2, 3, 4, 4, 4]
  arr2 = [1, 1, 2, 2, 3, 3, 4, 4, 4]
  @test Hgm.W1(arr1, arr2, normalize = false) == 0.25
  @test Hgm.W1(arr2, arr1, normalize = false) == 0.25
  @test Hgm.W1(arr1, arr2) == Hgm.W1(arr2, arr1)
  
  @test isapprox(Hgm.W1(x1_dns, x1_bal), w1_dns_bal)
  @test isapprox(Hgm.W1(x1_dns, x1_onl), w1_dns_onl)
  @test isapprox(Hgm.W1(x1_bal, x1_onl), w1_bal_onl)
  @test isapprox(Hgm.W1(x1_dns, x1_bal, normalize = false), w1_dns_bal_unnorm)

  @test size(Hgm.W1(rand(3,100), rand(3,100))) == (3,)
  @test size(Hgm.W1(rand(9,100), rand(3,100))) == (3,)
end
println("")


