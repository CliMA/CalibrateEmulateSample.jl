using Test
import DifferentialEquations
import NPZ

const DE = DifferentialEquations

include("../../examples/L96m/L96m.jl")

const RPAD = 25
const LPAD_INTEGER = 7
const LPAD_FLOAT = 13
l96 = L96m()

const data_dir = joinpath(@__DIR__, "data")
const z0 = NPZ.npzread(joinpath(data_dir, "ic.npy"))
const Yk_test = NPZ.npzread(joinpath(data_dir, "Yk.npy"))
const full_test = NPZ.npzread(joinpath(data_dir, "full.npy"))
const balanced_test = NPZ.npzread(joinpath(data_dir, "balanced.npy"))
const regressed_test = NPZ.npzread(joinpath(data_dir, "regressed.npy"))
const dp5_pairs = NPZ.npzread(joinpath(data_dir, "dp5_pairs.npy"))
const dp5_test = NPZ.npzread(joinpath(data_dir, "dp5_full.npy"))
const tp8_test = NPZ.npzread(joinpath(data_dir, "tsitpap8_full.npy"))
const bal_test = NPZ.npzread(joinpath(data_dir, "dp5_bal.npy"))

################################################################################
# unit testing #################################################################
################################################################################
@testset "unit testing" begin
  Yk_comp = compute_Yk(l96, z0)
  @test isapprox(Yk_test, Yk_comp, atol=1e-15)

  full_comp = similar(z0)
  full(full_comp, z0, l96, 0.0)
  @test isapprox(full_test, full_comp, atol=1e-15)

  balanced_comp = similar(z0[1:l96.K])
  balanced(balanced_comp, z0[1:l96.K], l96, 0.0)
  @test isapprox(balanced_test, balanced_comp, atol=1e-15)

  set_G0(l96)
  regressed_comp = similar(z0[1:l96.K])
  regressed(regressed_comp, z0[1:l96.K], l96, 0.0)
  @test isapprox(balanced_test, regressed_comp, atol=1e-13)
  set_G0(l96, slope = -0.7)
  regressed(regressed_comp, z0[1:l96.K], l96, 0.0)
  @test isapprox(regressed_test, regressed_comp, atol=1e-15)

  @testset "G0 testing" begin
    v1 = Vector(1.0:20.0)
    v2 = Vector(1:3:21)
    set_G0(l96)
    @test l96.hy * 5.0 == l96.G(5.0)
    @test l96.hy * v1 == l96.G(v1)
    @test l96.hy * v2 == l96.G(v2)
    set_G0(l96, 1.0)
    @test 5.0 == l96.G(5.0)
    @test v1 == l96.G(v1)
    @test v2 == l96.G(v2)
    set_G0(l96, slope = -0.5)
    @test -0.5 * v1 == l96.G(v1)
    @test -0.5 * v2 == l96.G(v2)
  end

  m1 = [1:9.0  1:9.0]
  z00 = zeros(l96.K + l96.K * l96.J)
  z00[1:l96.K] .= 1:l96.K
  for k_ in 1:l96.K
    z00[l96.K + 1 + (k_-1)*l96.J : l96.K + k_*l96.J] .= z00[k_]
  end
  @test isapprox(gather_pairs(l96, z00), m1)

  z00 = random_init(l96)
  x00 = @view(z00[1:l96.K])
  y00 = @view(z00[l96.K+1:end])
  @test size(z00,1) == l96.K + l96.K * l96.J
  @test size(z00,1) == length(z00)
  @test all(-5.0 .< x00 .< 10.0)
  @test all(x00 .== reshape(y00, l96.J, l96.K)')

  m2 = [x00  x00]
  @test isapprox(gather_pairs(l96, z00), m2)

  @test isapprox(gather_pairs(l96, dp5_test), dp5_pairs)
end
println("")

################################################################################
# integration testing ##########################################################
################################################################################
const T = 0.1
const dtmax = 1e-5
const saveat = 1e-2
pb_full = DE.ODEProblem(full, z0, (0.0, T), l96)
pb_bal = DE.ODEProblem(balanced, z0[1:l96.K], (0.0, T), l96)
pb_reg = DE.ODEProblem(regressed, z0[1:l96.K], (0.0, T), l96)

# full, DP5
print(rpad("(full, DP5)", RPAD))
elapsed_dp5 = @elapsed begin
  sol_dp5 = DE.solve(pb_full, DE.DP5(), dtmax = dtmax, saveat = saveat)
end
println("steps:", lpad(length(sol_dp5.t), LPAD_INTEGER),
        "\telapsed:", lpad(elapsed_dp5, LPAD_FLOAT))

# full, TsitPap8
print(rpad("(full, TsitPap8)", RPAD))
elapsed_tp8 = @elapsed begin
  sol_tp8 = DE.solve(pb_full, DE.TsitPap8(), dtmax = dtmax, saveat = saveat)
end
println("steps:", lpad(length(sol_tp8.t), LPAD_INTEGER),
        "\telapsed:", lpad(elapsed_tp8, LPAD_FLOAT))

# balanced, DP5
print(rpad("(balanced, DP5)", RPAD))
elapsed_bal = @elapsed begin
  sol_bal = DE.solve(pb_bal, DE.DP5(), dtmax = dtmax, saveat = saveat)
end
println("steps:", lpad(length(sol_bal.t), LPAD_INTEGER),
        "\telapsed:", lpad(elapsed_bal, LPAD_FLOAT))

# regressed, DP5
set_G0(l96) # this has a side effect on `pb_reg` through `l96` object
print(rpad("(regressed, DP5)", RPAD))
elapsed_reg = @elapsed begin
  sol_reg = DE.solve(pb_reg, DE.DP5(), dtmax = dtmax, saveat = saveat)
end
println("steps:", lpad(length(sol_reg.t), LPAD_INTEGER),
        "\telapsed:", lpad(elapsed_reg, LPAD_FLOAT))

@testset "integration testing" begin
  @test isapprox(sol_dp5[:,1:end], dp5_test, atol=1e-3)
  @test isapprox(sol_tp8[:,1:end], tp8_test, atol=1e-3)
  @test isapprox(sol_bal[:,1:end], bal_test, atol=1e-5)
  @test isapprox(sol_reg[:,1:end], bal_test, atol=1e-5) # reg + G0 == bal
end
println("")

