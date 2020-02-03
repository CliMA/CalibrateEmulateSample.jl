using Test

using CalibrateEmulateSample.Utilities

@testset "Utilities" begin
  @test length(name("a"))        == RPAD
  @test length(name("a" ^ RPAD)) == (RPAD + 1)
  @test length(warn("a"))        == RPAD
  @test length(warn("a" ^ RPAD)) == (RPAD + 11)
  @test isa(name("a"), String)
  @test isa(warn("a"), String)
end
