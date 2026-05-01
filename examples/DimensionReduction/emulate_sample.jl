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
# select the forward map
mod_types = ["linear"]
mod_type = mod_types[1]
@info "Executing model type $(mod_type)"

rng = Random.MersenneTwister(41)
input_dim = 100
output_dim = 100

num_trials = 1
for trial in 1:num_trials
    loaded = load("datafiles/ekp_$(mod_type)_$(trial).jld2")
    ekpobj = loaded["ekpobj"]
    ekp_samples = loaded["ekp_samples"]
    prior = loaded["prior_obj"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]
    true_parameter = loaded["true_parameter"]
    αs = loaded["alphas"]

    min_iter = 1
    max_iter = min(10, length(αs)) # number of EKP iterations to use data from is at most this
    training_points = [ekp_samples[α] for α in αs[min_iter:max_iter]]

    encoder_schedule_ref = [(decorrelate_structure_mat(; retain_var = 1.0), "in_and_out")]

    rvs = [0.9, 0.94, 0.98, 0.99, 0.995, 0.999]
    rkls = rvs
    all_errs = zeros(length(rvs), 1 + length(αs))

    # as we have more input samples than out stored in ekp. we provide the extended set of outputs to be used in the likelihood informed processor
    final_samples_out = ekp_samples[αs[end]][2]
    encoder_kwargs = encoder_kwargs_from(ekpobj, prior; final_samples_out = final_samples_out)

    flat_io_pairs = get_training_points(ekpobj, min_iter:max_iter, g_final = final_samples_out)

    names = ["reference", "decorrelate, PCA-in", ["decorrelate, LI-in 1:$(i)" for i in 1:length(αs)]...]

    if mod_type == "linear"
        ni_scaling = 0.01 # noise injection into null space scaling (def. 1)
    else
        bad_model(mod_type, mod_types)
    end

    reduced_dims = zeros(Int, length(rvs), length(αs) + 2) # store reduced dims for final table

    for (idx, (rv, rkl)) in enumerate(zip(rvs, rkls))
        encoder_schedule_decorrelate =
            [(decorrelate_structure_mat(retain_var = rv), "in"), (decorrelate_structure_mat(), "out")]
        encoder_schedules_li = [
            [(decorrelate_structure_mat(), "in_and_out"), (likelihood_informed(retain_info = rkl, iters = 1:i), "in")] for i in 1:length(αs)
        ]

        em_ref = forward_map_wrapper(
            param -> forward_map(param, model),
            prior,
            flat_io_pairs;
            encoder_schedule = encoder_schedule_ref,
            encoder_kwargs = deepcopy(encoder_kwargs),
            noise_injector_scaling = ni_scaling, # shouldnt be needed
        )

        em_decorrelate = forward_map_wrapper(
            param -> forward_map(param, model),
            prior,
            flat_io_pairs;
            encoder_schedule = encoder_schedule_decorrelate,
            encoder_kwargs = deepcopy(encoder_kwargs),
            noise_injector_scaling = ni_scaling,
        )

        ems_li = [
            forward_map_wrapper(
                param -> forward_map(param, model),
                prior,
                flat_io_pairs;
                encoder_schedule = encoder_schedule,
                encoder_kwargs = deepcopy(encoder_kwargs),
                noise_injector_scaling = ni_scaling,
            ) for encoder_schedule in encoder_schedules_li
        ]
        post_means = reshape(true_parameter, input_dim, 1)
        post_covs = []
        for (iidx, nn, em) in zip(1:length(names), names, vcat(em_ref, em_decorrelate, ems_li...))

            reduced_dims[idx, iidx] = get_encoded_dim(get_encoder_schedule(em), "in")
            println(" ")
            @info "Encoding name: $(nn)"
            E, _ = get_encoder_from_schedule(get_encoder_schedule(em), "in")

            if isnothing(E)
                @info "No truncation"
            else
                ed = get_encoded_dim(get_encoder_schedule(em), "in")
                @info "Truncation criteria (var>$(rv), or kl > $(rkl))"
                @info "input dim reduced to: $(ed)"

            end
            println(" ")
            u0 = rand(MvNormal(mean(prior), cov(prior)))
            mcmc = MCMCWrapper(RWMHSampling(), y, prior, em; init_params = u0)
            new_step = optimize_stepsize(mcmc; init_stepsize = 0.05, N = 2000, discard_initial = 0)

            println("Begin MCMC - with step size ", new_step)
            mcmc = MCMCWrapper(RWMHSampling(), y, prior, em; init_params = u0)
            chain = MarkovChainMonteCarlo.sample(mcmc, 40_000; stepsize = new_step, discard_initial = 5_000)
            posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

            post_mean = mean(posterior)
            post_cov = cov(posterior)

            post_means = hcat(post_means, reshape(post_mean, input_dim, 1))
            push!(post_covs, post_cov)
        end

        # post_means[:,1] = true parameter
        # post_means[:,2] = ref
        # post_means[:,3] = decor, LI etc.
        # so err cols are normalized diff to ref of (decor, LI etc.) (lower is better)
        all_errs[idx, :] = [norm(post_means[:, 2] - v) / norm(post_means[:, 2]) for v in eachcol(post_means[:, 3:end])]'
    end
    @info "error of posterior mean to whitened \"reference\" solution. for $(names[2:end])"
    display(all_errs)
    @info "... and their reduced dimensions"
    display(reduced_dims[:, 2:end])

end
