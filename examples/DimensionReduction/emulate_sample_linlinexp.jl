using Distributions
using EnsembleKalmanProcesses
using JLD2
using LinearAlgebra
using Random
using MCMCChains
using AdvancedMH

# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities

include("./models.jl")


mutable struct NoEmulation <: Emulators.MachineLearningTool
    f::Any
    output_structure_mats::Any

    NoEmulation(f) = new(f, nothing)
end
function Emulators.build_models!(ne::NoEmulation, iopairs, input_structure_mats, output_structure_mats; mlt_kwargs...)
    ne.output_structure_mats = output_structure_mats
end
function Emulators.predict(ne::NoEmulation, new_inputs; mlt_kwargs...)
    encoder_schedule = mlt_kwargs[:encoder_schedule]

    decoded_inputs = decode_with_schedule(encoder_schedule, EnsembleKalmanProcesses.DataContainer(new_inputs), "in")
    decoded_outputs = hcat(map(ne.f, eachcol(decoded_inputs.data))...)
    encoded_outputs =
        encode_with_schedule(encoder_schedule, EnsembleKalmanProcesses.DataContainer(decoded_outputs), "out").data
    encoded_outputs,
    cat([Utilities.get_structure_mat(ne.output_structure_mats) for _ in eachcol(new_inputs)]...; dims = 3)
end


rng = Random.MersenneTwister(41)
input_dim = 100
output_dim = 100
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
    true_parameter = loaded["true_parameter"]


    min_iter = 1
    max_iter = 10 # number of EKP iterations to use data from is at most this

    encoder_schedule_ref = [(decorrelate_structure_mat(; retain_var = 1.0), "in_and_out")]

    dims = [2, 4, 6, 8, 10]
    all_errs = zeros(length(dims), 1 + length(αs))
    for (dim_i, dim) in enumerate(dims)
        encoder_schedule_decorrelate = [
            (Decorrelator([], [], [], (:dimension, dim), "structure_mat", nothing), "in"),
            (decorrelate_structure_mat(; retain_var = 1.0), "out"),
        ]
        encoder_schedules_li = [
            [
                (decorrelate_structure_mat(; retain_var = 1.0), "in_and_out"),
                (LikelihoodInformed(nothing, nothing, nothing, (:dimension, dim), α, :linreg, false), "in"),
            ] for α in αs
        ]

        em_ref = Emulator(
            NoEmulation(param -> forward_map(param, model)),
            Utilities.get_training_points(ekpobj, min_iter:max_iter);
            encoder_schedule = encoder_schedule_ref,
            encoder_kwargs = (; prior_cov = cov(prior_obj), obs_noise_cov = obs_noise_cov),
        )

        em_decorrelate = Emulator(
            NoEmulation(param -> forward_map(param, model)),
            Utilities.get_training_points(ekpobj, min_iter:max_iter);
            encoder_schedule = encoder_schedule_decorrelate,
            encoder_kwargs = (; prior_cov = cov(prior_obj), obs_noise_cov = obs_noise_cov),
        )

        ems_li = [
            Emulator(
                NoEmulation(param -> forward_map(param, model)),
                Utilities.get_training_points(ekpobj, min_iter:max_iter);
                encoder_schedule = encoder_schedule,
                encoder_kwargs = (;
                    prior_cov = cov(prior_obj),
                    obs_noise_cov = obs_noise_cov,
                    samples_in = ekp_samp[α][1],
                    samples_out = ekp_samp[α][2],
                    observation = y,
                ),
            ) for (encoder_schedule, α) in zip(encoder_schedules_li, αs)
        ]

        post_means = reshape(true_parameter, input_dim, 1)
        post_covs = []
        for em in vcat(em_ref, em_decorrelate, ems_li...)
            u0 = rand(MvNormal(mean(prior_obj), cov(prior_obj)))
            mcmc = MCMCWrapper(RWMHSampling(), y, prior_obj, em; init_params = u0)
            new_step = optimize_stepsize(mcmc; init_stepsize = 0.05, N = 2000, discard_initial = 0)

            println("Begin MCMC - with step size ", new_step)
            num_chains = 16
            mcmc = MCMCWrapper(RWMHSampling(), y, prior_obj, em; init_params = [u0 for _ in 1:num_chains])
            chain = MarkovChainMonteCarlo.sample(
                mcmc,
                MCMCThreads(),
                40_000,
                num_chains;
                chain_type = Chains,
                stepsize = new_step,
                discard_initial = 5_000,
            )

            posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

            post_mean = mean(posterior)
            post_cov = cov(posterior)

            post_means = hcat(post_means, reshape(post_mean, input_dim, 1))
            push!(post_covs, post_cov)
        end

        all_errs[dim_i, :] =
            [norm(post_means[:, 2] - v) / norm(post_means[:, 2]) for v in eachcol(post_means[:, 3:end])]'
    end

    display(all_errs)
end
