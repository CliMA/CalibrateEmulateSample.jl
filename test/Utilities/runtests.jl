using Test
using Random
using Statistics
using LinearAlgebra

using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.Observations
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.DataContainers

@testset "Utilities" begin

    # Seed for pseudo-random number generator
    rng = Random.MersenneTwister(41)

    arr = vcat([i * ones(3)' for i in 1:5]...)
    arr_t = permutedims(arr, (2, 1))
    data_names = ["d1", "d2", "d3"]
    obs = Observation(arr_t, data_names) #data must be columns as default
    sample = get_obs_sample(rng, obs)
    @test sample == [5.0, 5.0, 5.0]

    mean_arr = dropdims(mean(arr, dims = 1), dims = 1)
    std_arr = dropdims(std(arr, dims = 1), dims = 1)
    @test mean_arr == [3.0, 3.0, 3.0]
    @test std_arr ≈ [1.58, 1.58, 1.58] atol = 1e-2
    z_arr = orig2zscore(arr, mean_arr, std_arr)
    z_arr_test = [
        -1.265 -1.265 -1.265
        -0.632 -0.632 -0.632
        0.0 0.0 0.0
        0.632 0.632 0.632
        1.265 1.265 1.265
    ]
    @test z_arr ≈ z_arr_test atol = 1e-2
    orig_arr = zscore2orig(z_arr, mean_arr, std_arr)
    @test orig_arr ≈ arr atol = 1e-5

    v = vec([1.0 2.0 3.0])
    mean_v = vec([0.5 1.5 2.5])
    std_v = vec([0.5 0.5 0.5])
    z_v = orig2zscore(v, mean_v, std_v)
    println(z_v)
    @test z_v ≈ [1.0, 1.0, 1.0] atol = 1e-5
    orig_v = zscore2orig(z_v, mean_v, std_v)
    @test orig_v ≈ v atol = 1e-5

    # test get_training_points
    # first create the EnsembleKalmanProcess
    n_ens = 10
    dim_obs = 3
    dim_par = 2
    initial_ensemble = randn(rng, dim_par, n_ens)#params are cols
    y_obs = randn(rng, dim_obs)
    Γy = Matrix{Float64}(I, dim_obs, dim_obs)
    ekp = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), rng = rng)
    g_ens = randn(rng, dim_obs, n_ens) # data are cols
    EnsembleKalmanProcesses.update_ensemble!(ekp, g_ens)
    training_points = get_training_points(ekp, 1)
    @test get_inputs(training_points) ≈ initial_ensemble
    @test get_outputs(training_points) ≈ g_ens

end
