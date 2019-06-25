using Test
import NPZ
import LinearAlgebra

include("../../src/GPRWrap/GPRWrap.jl")

const data_dir = joinpath(@__DIR__, "data")
const gpr_mesh = NPZ.npzread(joinpath(data_dir, "mesh.npy"))
const gpr_mean = NPZ.npzread(joinpath(data_dir, "mean.npy"))
const gpr_subsample = NPZ.npzread(joinpath(data_dir, "subsample.npy"))

const inf_norm = x -> LinearAlgebra.norm(x, Inf)

################################################################################
# unit testing #################################################################
################################################################################
X = 0:0.1:1
y = X.^2
xmesh = 0:0.01:1
gprw = GPRWrap()

@testset "struct testing" begin
  @test !gprw.__data_set
  @test !gprw.__subsample_set
  @test gprw.data == nothing
  @test gprw.subsample == nothing
  @test gprw.GPR == nothing
end
println("")

thrsh = gprw.thrsh
@testset "methods testing" begin
  ex_thrown1 = false
  try
    subsample!(gprw)
  catch
    ex_thrown1 = true
  end
  @test ex_thrown1

  ex_thrown2 = false
  try
    set_data!(gprw, rand(20))
  catch
    ex_thrown2 = true
  end
  @test ex_thrown2

  set_data!(gprw, rand(20,2,2))
  @test gprw.__data_set
  @test !gprw.__subsample_set
  @test size(gprw.data,1) == 20

  set_data!(gprw, cat(X,y,dims=2))
  @test gprw.__data_set
  @test !gprw.__subsample_set

  ex_thrown3 = false
  try
    learn!(gprw)
  catch
    ex_thrown3 = true
  end
  @test !ex_thrown3
  @test gprw.__subsample_set

  learn!(gprw, kernel = "matern")
  @test occursin(r"matern"i, string(gprw.GPR.kernel))

  set_data!(gprw, rand(20,2))
  @test !gprw.__subsample_set
  @test gprw.GPR == nothing
  subsample!(gprw, indices = [1,2,3])
  @test gprw.__subsample_set
  subsample!(gprw, -1)
  @test size(gprw.subsample,1) == 20
  subsample!(gprw)
  @test size(gprw.subsample,1) == 20

  set_data!(gprw, cat(X,y,dims=2))
  learn!(gprw, alpha = 1e-10)
  @test occursin(r"rbf"i, string(gprw.GPR.kernel))

  ytrue = xmesh.^2
  ypred = predict(gprw, xmesh)
  @test isapprox(ytrue, ypred, atol=1e-5, norm=inf_norm)

  @test gprw.thrsh == thrsh
end
println("")

################################################################################
# non-synthetic data testing ###################################################
################################################################################
set_data!(gprw, gpr_subsample)
gprw.thrsh = -1
learn!(gprw)
@testset "non-synthetic testing" begin
  @test size(gprw.data,1) == 800
  @test size(gprw.subsample,1) == 800
  @test isapprox( predict(gprw, gpr_mesh), gpr_mean, atol=1e-3, norm=inf_norm)
end
println("")

