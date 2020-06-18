using Test
using Random
using Statistics

using CalibrateEmulateSample.Observations

@testset "Observations" begin

    # Generate samples as vector of vectors
    n_data = 3
    n_samples = 5
    samples = [vec(i*ones(n_data)) for i in 1:n_samples]
    data_names = ["d$(string(i))" for i in 1:n_data]
    obs = Obs(samples, data_names)
    @test obs.data_names == data_names
    @test obs.mean == [3.0, 3.0, 3.0]
    @test obs.cov == 2.5 * ones(3, 3)

    # Generate a single sample
    sample = [1.0, 2.0, 3.0]
    obs = Obs(sample, data_names)
    @test obs.mean == sample
    @test obs.cov == nothing

    # Generate samples as a 2d-array (each row corresponds to 1 sample)
    samples = vcat([i*ones(n_data)' for i in 1:n_samples]...)
    obs = Obs(samples, data_names)
    @test obs.mean == [3.0, 3.0, 3.0]
    @test obs.cov == 2.5 * ones(3, 3)

end


