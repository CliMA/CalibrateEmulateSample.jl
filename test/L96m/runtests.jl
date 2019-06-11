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
const dp5_test = NPZ.npzread(joinpath(data_dir, "dp5_full.npy"))
const tp8_test = NPZ.npzread(joinpath(data_dir, "tsitpap8_full.npy"))
const bal_test = NPZ.npzread(joinpath(data_dir, "dp5_bal.npy"))

################################################################################
# unit testing #################################################################
################################################################################
Yk_comp = compute_Yk(l96, z0)
full_comp = similar(z0)
full(full_comp, z0, l96, 0.0)
balanced_comp = similar(z0[1:l96.K])
balanced(balanced_comp, z0[1:l96.K], l96, 0.0)
@testset "unit testing" begin
  @test isapprox(Yk_test, Yk_comp, atol=1e-15)
  @test isapprox(full_test, full_comp, atol=1e-15)
  @test isapprox(balanced_test, balanced_comp, atol=1e-15)
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

@testset "integration testing" begin
  @test isapprox(sol_dp5[:,1:end], dp5_test, atol=1e-3)
  @test isapprox(sol_tp8[:,1:end], tp8_test, atol=1e-3)
  @test isapprox(sol_bal[:,1:end], bal_test, atol=1e-5)
end
println("")

