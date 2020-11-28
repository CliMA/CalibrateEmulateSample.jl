using Test
using Distributions
using StatsBase
using Random

using CalibrateEmulateSample.ParameterDistributionStorage

@testset "ParameterDistribution" begin
    @testset "ParameterDistributionType" begin
        # Tests for the ParameterDistributionType
        d = Parameterized(Gamma(2.0, 0.8))
        @test d.distribution == Gamma(2.0,0.8)
        
        d = Samples([1 2 3; 4 5 6])
        @test d.distribution_samples == [1 2 3; 4 5 6]
        d = Samples([1 4; 2 5; 3 6]; params_are_columns=false)
        @test d.distribution_samples == [1 2 3; 4 5 6]
        d = Samples([1, 2, 3, 4, 5, 6])
        @test d.distribution_samples == [1 2 3 4 5 6]
        d = Samples([1, 2, 3, 4, 5, 6]; params_are_columns=false)
        @test d.distribution_samples == reshape([1, 2, 3, 4, 5, 6],:,1)
        
    end
    @testset "ConstraintType" begin
        # Tests for the ConstraintType
        c = BoundedBelow(0.2)
        @test c.lower_bound == 0.2
        
        c = BoundedAbove(0.2)
        @test c.upper_bound == 0.2
        
        c = Bounded(-0.1,0.2)
        @test c.lower_bound == -0.1
        @test c.upper_bound == 0.2
        @test_throws DomainError Bounded(0.2,-0.1)
    end

    @testset "ParameterDistribution(s)" begin

        d = Parameterized(Normal(0,1))
        c = NoConstraint()
        name = "unconstrained_normal"
        u = ParameterDistribution(d,c,name)
        @test u.distributions == [d]
        @test u.constraints == [c]
        @test u.names == [name]

        # Tests for the ParameterDistribution
        d = Parameterized(MvNormal(4,0.1))
        c = [NoConstraint(),
             BoundedBelow(-1.0),
             BoundedAbove(0.4),
             Bounded(-0.1,0.2)]
      
        name = "constrained_mvnormal"
        u = ParameterDistribution(d,c,name)
        @test u.distributions == [d]
        @test u.constraints == c
        @test u.names == [name]
        @test_throws DimensionMismatch ParameterDistribution(d,c[1:3],name)
        
        # Tests for the ParameterDistribution
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [NoConstraint(),
              BoundedBelow(-1.0),
              BoundedAbove(0.4),
              Bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [Bounded(10,15),
              NoConstraint()]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistribution([d1,d2], [c1,c2], [name1,name2])
        @test u.distributions == [d1,d2]
        @test u.constraints == cat([c1,c2]...,dims=1)
        @test u.names == [name1,name2]
    end

    @testset "get/sample functions" begin
        # setup for the tests:
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [NoConstraint(),
              BoundedBelow(-1.0),
              BoundedAbove(0.4),
              Bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([1 2 3 4])
        c2 = [Bounded(10,15)]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistribution([d1,d2], [c1,c2], [name1,name2])
        
        # Tests for get_name
        @test get_name(u1) == [name1]
        @test get_name(u) == [name1, name2]
        
        # Tests for get_distribution
        @test get_distribution(d1) == MvNormal(4,0.1)
        @test get_distribution(u1)[name1] == MvNormal(4,0.1)
        @test typeof(get_distribution(d2)) <: String
        @test typeof(get_distribution(u2)[name2]) <: String
        
        d = get_distribution(u)
        @test d[name1] == MvNormal(4,0.1)
        @test typeof(d[name2]) <: String

        # Tests for sample distribution
        seed=2020
        Random.seed!(seed)
        s1 = rand(MvNormal(4,0.1),1)
        Random.seed!(seed)
        @test sample_distribution(u1) == s1

        Random.seed!(seed)
        s1 = rand(MvNormal(4,0.1),3)
        Random.seed!(seed)
        @test sample_distribution(u1,3) == s1
        
        Random.seed!(seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]),1) 
        s2 = d2.distribution_samples[:, idx]
        Random.seed!(seed)
        @test sample_distribution(u2) == s2

        Random.seed!(seed)
        idx = StatsBase.sample(collect(1:size(d2.distribution_samples)[2]), 3) 
        s2 = d2.distribution_samples[:, idx]
        Random.seed!(seed)
        @test sample_distribution(u2,3) == s2
        
        Random.seed!(seed)
        s1 = sample_distribution(u1,3)
        s2 = sample_distribution(u2,3)
        Random.seed!(seed)
        s = sample_distribution(u,3)
        @test s == cat([s1,s2]...,dims=1)
    end

    @testset "transform functions" begin
        #setup for the tests
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [NoConstraint(),
              BoundedBelow(-1.0),
              BoundedAbove(0.4),
              Bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([1.0 3.0 5.0 7.0; 9.0 11.0 13.0 15.0])
        c2 = [Bounded(10,15),
              NoConstraint()]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistribution([d1,d2], [c1,c2], [name1,name2])

        x_unbd = rand(MvNormal(6,3), 1000)  #6 x 1000 
        # Tests for transforms
        # prior to real
        x_real_noconstraint = map(x -> transform_prior_to_real(NoConstraint(), x), x_unbd[1,:])
        @test x_real_noconstraint == x_unbd[1,:]
        x_real_below = map(x -> transform_prior_to_real(BoundedBelow(30), x), x_unbd[1,:])
        @test all(x_real_below .> 30)
        x_real_above = map(x -> transform_prior_to_real(BoundedAbove(-10), x), x_unbd[1,:])
        @test all(x_real_above .< -10)
        x_real_bounded = map(x -> transform_prior_to_real(Bounded(-2,-1), x), x_unbd[1,:])
        @test all([all(x_real_bounded .> -2), all(x_real_bounded .< -1)])

        # prior to real
        @test isapprox(x_unbd[1,:] - map(x -> transform_real_to_prior(NoConstraint(), x), x_real_noconstraint), zeros(size(x_unbd)[2]); atol=1e-6)
        @test isapprox(x_unbd[1,:] - map(x -> transform_real_to_prior(BoundedBelow(30), x), x_real_below), zeros(size(x_unbd)[2]) ; atol=1e-6) 
        @test isapprox(x_unbd[1,:] - map(x -> transform_real_to_prior(BoundedAbove(-10), x), x_real_above), zeros(size(x_unbd)[2]); atol=1e-6)
        @test isapprox(x_unbd[1,:] - map(x -> transform_real_to_prior(Bounded(-2,-1), x), x_real_bounded), zeros(size(x_unbd)[2]); atol=1e-6)

        x_real_constrained1 = mapslices(x -> transform_prior_to_real(u1,x), x_unbd[1:4,:]; dims=1)
        @test isapprox(x_unbd[1:4,:] - mapslices(x -> transform_real_to_prior(u1,x), x_real_constrained1; dims=1), zeros(size(x_unbd[1:4,:])) ; atol=1e-6)
        x_real_constrained2 = mapslices(x -> transform_prior_to_real(u2,x), x_unbd[5:6,:]; dims=1) 
        @test isapprox(x_unbd[5:6,:] -  mapslices(x -> transform_real_to_prior(u2,x), x_real_constrained2; dims=1), zeros(size(x_unbd[5:6,:])); atol=1e-6)
        
        x_real = mapslices(x -> transform_prior_to_real(u,x), x_unbd; dims=1)
        x_unbd_tmp = mapslices(x -> transform_real_to_prior(u,x), x_real; dims=1)
        @test isapprox(x_unbd - x_unbd_tmp,zeros(size(x_unbd));atol=1e-6)
        
    end
    
end
