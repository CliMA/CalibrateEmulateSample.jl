using Distributions
using EnsembleKalmanProcesses
using JLD2
using LinearAlgebra
using Random

# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities

include("./models.jl")

rng = Random.MersenneTwister(123)
input_dim = 100
output_dim = 50
αs = [0.0, 0.25, 0.5, 0.75, 1.0]

num_trials = 1
for trial in 1:num_trials
    loaded = load("datafiles/ekp_linlinexp_$(trial).jld2")
    ekpobj = loaded["ekpobj"]
    ekp_samp = loaded["ekp_samples"]
    prior_obj = loaded["prior_obj"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]


    min_iter = 1
    max_iter = 7 # number of EKP iterations to use data from is at most this

    encoder_schedule_decorrelate = [(decorrelate_structure_mat(; retain_var = 0.7), "in_and_out")]
    encoder_schedules_li = [
        [(likelihood_informed(; retain_KL = 0.9, alpha = α, use_data_as_samples = false), "in_and_out")] for α in αs
    ]

    em_decorrelate = Emulator(
        GaussianProcess(Emulators.GPJL(); kernel = nothing, prediction_type = Emulators.YType(), noise_learn = false),
        Utilities.get_training_points(ekpobj, min_iter:max_iter);
        encoder_schedule = encoder_schedule_decorrelate,
        encoder_kwargs = (; prior_cov = cov(prior_obj), obs_noise_cov = obs_noise_cov),
    )

    ems_li = [
        Emulator(
            GaussianProcess(Emulators.GPJL(); kernel = nothing, prediction_type = Emulators.YType(), noise_learn = false),
            Utilities.get_training_points(ekpobj, min_iter:max_iter);
            encoder_schedule = encoder_schedule,
            encoder_kwargs = (; prior_cov = cov(prior_obj), obs_noise_cov = obs_noise_cov, samples_in = ekp_samp[α][1], samples_out = ekp_samp[α][2], observation = y),
        )
        for (encoder_schedule, α) in zip(encoder_schedules_li, αs)
    ]

    post_means = zeros(input_dim, 0)
    for em in vcat(em_decorrelate, ems_li...)
        u0 = rand(MvNormal(mean(prior_obj), cov(prior_obj)))
        mcmc = MCMCWrapper(RWMHSampling(), y, prior_obj, em; init_params = u0)
        new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

        println("Begin MCMC - with step size ", new_step)
        chain = MarkovChainMonteCarlo.sample(mcmc, 10_000; stepsize = new_step, discard_initial = 2_000)

        posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

        post_mean = mean(posterior)
        post_cov = cov(posterior)
        println("post_mean")
        println(post_mean)
        println("post_cov")
        println(post_cov)

        post_means = hcat(post_means, reshape(post_mean, input_dim, 1))
    end

    println(post_means[1:10,:])
end
