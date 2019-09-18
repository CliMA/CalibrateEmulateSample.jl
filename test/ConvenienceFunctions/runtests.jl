using Test

include("../../src/ConvenienceFunctions.jl")

################################################################################
# unit testing #################################################################
################################################################################
@testset "unit testing" begin
  @test isdefined(Main, :RPAD)
  @test length(name("a"))            == RPAD
  @test length(name("a" ^ RPAD))     == (RPAD + 1)
  @test length(warn("a"))            == RPAD
  @test length(warn("a" ^ RPAD))     == (RPAD + 11)
  @test isa(name("a"), String)
  @test isa(warn("a"), String)
end
println("")


