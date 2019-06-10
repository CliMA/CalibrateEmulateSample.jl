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
const dp5_test = NPZ.npzread(joinpath(data_dir, "dp5.npy"))
const tp8_test = NPZ.npzread(joinpath(data_dir, "tsitpap8.npy"))

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
const T = 4
pb = DE.ODEProblem(full, z0, (0.0, T), l96)

print(rpad("(full, DP5)", RPAD))
elapsed_dp5 = @elapsed begin
  sol_dp5 = DE.solve(pb, DE.DP5(), reltol=1e-5, abstol=1e-9, saveat=0.1)
end
println("steps:", lpad(length(sol_dp5.t), LPAD_INTEGER),
        "\telapsed:", lpad(elapsed_dp5, LPAD_FLOAT))

print(rpad("(full, TsitPap8)", RPAD))
elapsed_tp8 = @elapsed begin
  sol_tp8 = DE.solve(pb, DE.TsitPap8(), reltol=1e-5, abstol=1e-9, saveat=0.1)
end
println("steps:", lpad(length(sol_tp8.t), LPAD_INTEGER),
        "\telapsed:", lpad(elapsed_tp8, LPAD_FLOAT))

@testset "integration testing" begin
  @test isapprox(sol_dp5[:,1:end], dp5_test, atol=5e-8)
  @test isapprox(sol_tp8[:,1:end], tp8_test, atol=5e-8)
end
println("")

