using Test
using Distributions

using CalibrateEmulateSample.Priors

@testset "Priors" begin

    # u1 is a parameter with a Gamma prior
    u1_prior = Prior(Gamma(2.0, 0.8), "u1")
    @test u1_prior.param_name == "u1"
    @test u1_prior.dist == Gamma(2.0, 0.8)

    # u2 is a parameter with a Deterministic prior, i.e., it always takes on
    # the same value, val
    val = 3.0
    u2_prior = Prior(Deterministic(val), "u2")
    @test u2_prior.param_name == "u2"
    @test u2_prior.dist == Deterministic(val)
    @test u2_prior.dist.value == val

end
