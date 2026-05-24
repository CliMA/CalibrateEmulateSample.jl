using Test
using LinearAlgebra
using Distributions
using GaussianProcesses

using CalibrateEmulateSample
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.ParameterDistributions
import AdvancedMH
import AbstractMCMC

const MCMC = MarkovChainMonteCarlo

@testset "Show" begin

    # ── Utilities ──────────────────────────────────────────────────────────────

    @testset "ElementwiseScaler" begin
        es = quartile_scale()
        out = sprint(show, MIME("text/plain"), es)
        @test occursin("ElementwiseScaler", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, es)
        @test occursin("ElementwiseScaler", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), es; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, es)
        @test occursin("ElementwiseScaler", outs)
        @test !occursin('\n', outs)
    end

    @testset "Decorrelator" begin
        dd = decorrelate_sample_cov()
        out = sprint(show, MIME("text/plain"), dd)
        @test occursin("Decorrelator", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, dd)
        @test occursin("Decorrelator", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), dd; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, dd)
        @test occursin("Decorrelator", outs)
        @test !occursin('\n', outs)
    end

    @testset "CanonicalCorrelation" begin
        cc = canonical_correlation()
        out = sprint(show, MIME("text/plain"), cc)
        @test occursin("CanonicalCorrelation", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, cc)
        @test occursin("CanonicalCorrelation", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), cc; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, cc)
        @test occursin("CanonicalCorrelation", outs)
        @test !occursin('\n', outs)
    end

    @testset "LikelihoodInformed" begin
        li = likelihood_informed()
        out = sprint(show, MIME("text/plain"), li)
        @test occursin("LikelihoodInformed", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, li)
        @test occursin("LikelihoodInformed", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), li; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, li)
        @test occursin("LikelihoodInformed", outs)
        @test !occursin('\n', outs)
    end

    @testset "NoiseInjector" begin
        ni = NoiseInjector(rand(3, 2), rand(2, 1), rand(3, 1), nothing, 0.5, true, [])
        out = sprint(show, MIME("text/plain"), ni)
        @test occursin("NoiseInjector", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, ni)
        @test occursin("NoiseInjector", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), ni; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, ni)
        @test occursin("NoiseInjector", outs)
        @test !occursin('\n', outs)
    end

    # ── Emulators ──────────────────────────────────────────────────────────────

    @testset "GaussianProcess" begin
        gp = GaussianProcess(GPJL())
        out = sprint(show, MIME("text/plain"), gp)
        @test occursin("GaussianProcess", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, gp)
        @test occursin("GaussianProcess", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), gp; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, gp)
        @test occursin("GaussianProcess", outs)
        @test !occursin('\n', outs)
    end

    @testset "ScalarRandomFeatureInterface" begin
        srfi = ScalarRandomFeatureInterface(100, 3)
        out = sprint(show, MIME("text/plain"), srfi)
        @test occursin("ScalarRandomFeatureInterface", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, srfi)
        @test occursin("ScalarRandomFeatureInterface", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), srfi; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, srfi)
        @test occursin("ScalarRandomFeatureInterface", outs)
        @test !occursin('\n', outs)
    end

    @testset "VectorRandomFeatureInterface" begin
        vrfi = VectorRandomFeatureInterface(100, 3, 2)
        out = sprint(show, MIME("text/plain"), vrfi)
        @test occursin("VectorRandomFeatureInterface", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, vrfi)
        @test occursin("VectorRandomFeatureInterface", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), vrfi; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, vrfi)
        @test occursin("VectorRandomFeatureInterface", outs)
        @test !occursin('\n', outs)
    end

    @testset "Emulator" begin
        n, p, d = 20, 3, 2
        iop = PairedDataContainer(rand(p, n), rand(d, n), data_are_columns = true)
        gp = GaussianProcess(GPJL())
        em = Emulator(gp, iop; encoder_schedule = [])
        out = sprint(show, MIME("text/plain"), em)
        @test occursin("Emulator", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, em)
        @test occursin("Emulator", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), em; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, em)
        @test occursin("Emulator", outs)
        @test !occursin('\n', outs)
    end

    @testset "ForwardMapWrapper" begin
        prior = constrained_gaussian("x", 0.0, 1.0, -Inf, Inf; repeats = 3)
        iop   = PairedDataContainer(rand(3, 5), rand(2, 5), data_are_columns = true)
        ni    = NoiseInjector(rand(2, 3), rand(3, 1), rand(2, 1), nothing, 0.5, true, [])
        fmw   = ForwardMapWrapper{Float64, Vector{Any}, typeof(prior), typeof(ni)}(
            identity, prior, iop, iop, [], ni,
        )
        out = sprint(show, MIME("text/plain"), fmw)
        @test occursin("ForwardMapWrapper", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, fmw)
        @test occursin("ForwardMapWrapper", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), fmw; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, fmw)
        @test occursin("ForwardMapWrapper", outs)
        @test !occursin('\n', outs)
    end

    # ── MarkovChainMonteCarlo ──────────────────────────────────────────────────

    @testset "RWMetropolisHastings" begin
        prop = AdvancedMH.RandomWalkProposal(MvNormal(zeros(2), I(2)))
        rw   = MCMC.RWMetropolisHastings{typeof(prop), GradFreeProtocol}(prop)
        out = sprint(show, MIME("text/plain"), rw)
        @test occursin("RWMetropolisHastings", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, rw)
        @test occursin("RWMetropolisHastings", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), rw; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, rw)
        @test occursin("RWMetropolisHastings", outs)
        @test !occursin('\n', outs)
    end

    @testset "pCNMetropolisHastings" begin
        prop = AdvancedMH.RandomWalkProposal(MvNormal(zeros(2), I(2)))
        pcn  = MCMC.pCNMetropolisHastings{typeof(prop), GradFreeProtocol}(prop)
        out = sprint(show, MIME("text/plain"), pcn)
        @test occursin("pCNMetropolisHastings", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, pcn)
        @test occursin("pCNMetropolisHastings", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), pcn; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, pcn)
        @test occursin("pCNMetropolisHastings", outs)
        @test !occursin('\n', outs)
    end

    @testset "BarkerMetropolisHastings" begin
        prop  = AdvancedMH.RandomWalkProposal(MvNormal(zeros(2), I(2)))
        barkr = MCMC.BarkerMetropolisHastings{typeof(prop), ForwardDiffProtocol}(prop)
        out = sprint(show, MIME("text/plain"), barkr)
        @test occursin("BarkerMetropolisHastings", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, barkr)
        @test occursin("BarkerMetropolisHastings", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), barkr; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, barkr)
        @test occursin("BarkerMetropolisHastings", outs)
        @test !occursin('\n', outs)
    end

    @testset "MCMCWrapper" begin
        prior      = constrained_gaussian("x", 0.0, 1.0, -Inf, Inf; repeats = 2)
        obs        = [rand(2) for _ in 1:3]
        log_post   = AdvancedMH.DensityModel(x -> -0.5 * sum(x .^ 2))
        prop       = AdvancedMH.RandomWalkProposal(MvNormal(zeros(2), I(2)))
        rw_sampler = MCMC.RWMetropolisHastings{typeof(prop), GradFreeProtocol}(prop)
        mcmcw = MCMCWrapper{typeof(obs), typeof(obs), Vector{Any}}(
            prior, prior, obs, obs, log_post, rw_sampler, (;), [],
        )
        out = sprint(show, MIME("text/plain"), mcmcw)
        @test occursin("MCMCWrapper", out)
        @test count(==('\n'), out) <= 10
        out2 = sprint(show, mcmcw)
        @test occursin("MCMCWrapper", out2)
        @test !occursin('\n', out2)
        out3 = sprint(show, MIME("text/plain"), mcmcw; context = :compact => true)
        @test out2 == out3
        outs = sprint(summary, mcmcw)
        @test occursin("MCMCWrapper", outs)
        @test !occursin('\n', outs)
    end

end
