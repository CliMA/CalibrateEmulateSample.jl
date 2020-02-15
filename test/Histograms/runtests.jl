using Test
using NPZ

using CalibrateEmulateSample
using CalibrateEmulateSample.Histograms
const Hgm = Histograms

const data_dir = joinpath(@__DIR__, "data")
const x1_bal = NPZ.npzread(joinpath(data_dir, "x1_bal.npy"))
const x1_dns = NPZ.npzread(joinpath(data_dir, "x1_dns.npy"))
const x1_onl = NPZ.npzread(joinpath(data_dir, "x1_onl.npy"))
const x2_bal = NPZ.npzread(joinpath(data_dir, "x2_bal.npy"))
const x2_onl = NPZ.npzread(joinpath(data_dir, "x2_onl.npy"))
# x2_dns is not needed

const w1_dns_bal        = 0.8190688772401341
const w1_dns_onl        = 0.10026156568190529
const w1_bal_onl        = 0.8280467119588161
const w1_dns_bal_normed = 0.03755967829782972
const w1_dns_bal_comb   = 0.818516185308166
const w1_bal_onl_x2     = 0.8781673689588835
const w1_bal_onl_comb   = 0.8529846357976019

@testset "Histograms: W1" begin
  arr1 = [1, 1, 1, 2, 3, 4, 4, 4]
  arr2 = [1, 1, 2, 2, 3, 3, 4, 4, 4]
  @test Hgm.W1(arr1, arr2; normalize = false) == 0.25
  @test Hgm.W1(arr2, arr1; normalize = false) == 0.25
  @test Hgm.W1([arr1; 5], arr2; normalize = true) ==
        Hgm.W1(arr2, [arr1; 5]; normalize = true)

  @test isapprox(Hgm.W1(x1_dns, x1_bal), w1_dns_bal)
  @test isapprox(Hgm.W1(x1_dns, x1_onl), w1_dns_onl)
  @test isapprox(Hgm.W1(x1_bal, x1_onl), w1_bal_onl)
  @test isapprox(Hgm.W1(x1_dns, x1_bal, normalize = true), w1_dns_bal_normed)

  @test size(Hgm.W1(rand(3,100), rand(3,100))) == (3,)
  @test size(Hgm.W1(rand(9,100), rand(3,100))) == (3,)
end

@testset "Histograms: HistData" begin
  FT = typeof(1.0)
  hd = Hgm.HistData(FT)
  Hgm.load!(hd, :unif,  rand(3,100000))
  Hgm.load!(hd, :norm, randn(3,100000))
  w1_rand = Hgm.W1(hd, :unif, :norm)
  @test size(w1_rand) == (3,)
  @test all(isapprox.(w1_rand, 0.7; atol = 3e-2))

  Hgm.load!(hd, :unif,  rand(3,10))
  Hgm.load!(hd, :norm, randn(1000))
  @test size(hd.samples[:unif], 1) == 3
  @test ndims(hd.samples[:norm]) == 1

  delete!(hd.samples, :unif)
  delete!(hd.samples, :norm)

  Hgm.load!(hd, :dns, joinpath(data_dir, "x1_dns.npy"))
  Hgm.load!(hd, :bal, [x1_bal'; x2_bal'])
  Hgm.load!(hd, :onl, [x1_onl'; x2_onl'])
  @test haskey(hd.samples, :dns)
  @test haskey(hd.samples, :onl)
  @test ndims(hd.samples[:dns]) == 1
  @test size(hd.samples[:onl], 1) == 2

  k2a_vectorized = Hgm.W1(hd, :bal)
  @test isa(k2a_vectorized[:dns], Real)
  @test size(k2a_vectorized[:onl]) == (2,)
  @test isapprox(k2a_vectorized[:dns], w1_dns_bal_comb)
  @test isapprox(k2a_vectorized[:onl], [w1_bal_onl, w1_bal_onl_x2])

  @test isapprox(Hgm.W1(hd, :bal, :onl, 1:2), w1_bal_onl_comb)
  @test isapprox(Hgm.W1(hd, :dns, :bal, 1:2), w1_dns_bal_comb)
end


