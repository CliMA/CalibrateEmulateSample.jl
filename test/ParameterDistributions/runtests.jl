using Test
using Distributions
using StatsBase
using Random

using CalibrateEmulateSample.ParameterDistributionsStorage

@testset "ParameterDistributions" begin
    @testset "ParameterDistributionType" begin
        # Tests for the ParameterDistributionType
        d = Parameterized(Gamma(2.0, 0.8))
        @test d.distribution == Gamma(2.0,0.8)
    
        d = Samples([[1 2 3]; [4 5 6]])
        @test d.distribution_samples == [[1.0 2.0 3.0]; [4.0 5.0 6.0]]
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
        # Tests for the ParameterDistribution
        d = Parameterized(MvNormal(4,0.1))
        c = [NoConstraint(),
             BoundedBelow(-1.0),
             BoundedAbove(0.4),
             Bounded(-0.1,0.2)]
      
        name = "constrained_mvnormal"
        u = ParameterDistribution(d,c,name)
        @test u.distribution == d
        @test u.constraints == c
        @test u.name == name
        @test_throws DimensionMismatch ParameterDistribution(d,c[1:3],name)

        # Tests for the ParameterDistribuions
        d1 = Parameterized(MvNormal(4,0.1))
        c1 = [NoConstraint(),
              BoundedBelow(-1.0),
              BoundedAbove(0.4),
              Bounded(-0.1,0.2)]
        name1 = "constrained_mvnormal"
        u1 = ParameterDistribution(d1,c1,name1)
        
        d2 = Samples([[1.0 3.0]; [5.0 7.0]; [9.0 11.0]; [13.0 15.0]])
        c2 = [Bounded(10,15),
              NoConstraint()]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistributions([u1,u2])
        @test u.parameter_distributions == [u1,u2]
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
        
        d2 = Samples([[1.0 3.0]; [5.0 7.0]; [9.0 11.0]; [13.0 15.0]])
        c2 = [Bounded(10,15),
              NoConstraint()]
        name2 = "constrained_sampled"
        u2 = ParameterDistribution(d2,c2,name2)
        
        u = ParameterDistributions([u1,u2])

        # Tests for get_name
        @test get_name(u1) == name1
        @test get_name(u) == [name1, name2]
        
        # Tests for get_distribution
        @test get_distribution(d1) == MvNormal(4,0.1)
        @test get_distribution(u1) == MvNormal(4,0.1)
        @test typeof(get_distribution(d2)) <: String
        @test typeof(get_distribution(u2)) <: String
        
        d = get_distribution(u)
        @test d[name1] == MvNormal(4,0.1)
        @test typeof(d[name2]) <: String

        # Tests for sample distribution
        Random.seed!(2020)
        s1 = rand(MvNormal(4,0.1),1)
        Random.seed!(2020)
        @test sample_distribution(u1) == s1

        Random.seed!(2020)
        s1 = rand(MvNormal(4,0.1),3)
        Random.seed!(2020)
        @test sample_distribution(u1,3) == s1
        
        Random.seed!(2020)
        idx = sample(collect(1:size(d2.distribution_samples)[1]),1) 
        s2 = d2.distribution_samples[idx, :]
        Random.seed!(2020)
        @test sample_distribution(u2) == s2

        Random.seed!(2020)
        idx = sample(collect(1:size(d2.distribution_samples)[1]), 3; replace=false) 
        s2 = d2.distribution_samples[idx, :]
        Random.seed!(2020)
        @test sample_distribution(u2,3) == s2
        
        Random.seed!(2020)
        s1 = sample_distribution(u1,3)
        s2 = sample_distribution(u2,3)
        Random.seed!(2020)
        s = sample_distribution(u,3)
        @test s[name1] == s1
        @test s[name2] == s2
    end

    @testset "transform functions" begin
        # Tests for transforms
    end
    
end
