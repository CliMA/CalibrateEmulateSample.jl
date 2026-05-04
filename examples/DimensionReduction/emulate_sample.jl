using Distributions
using EnsembleKalmanProcesses
using JLD2
using LinearAlgebra
using Random
using MCMCChains
using AdvancedMH
using DataFrames # table print at end 
# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities


include("./models.jl")
# select the forward map
mod_types = ["linear"]
mod_type = mod_types[1]
@info "Executing model type $(mod_type)"

function main()
    loaded = load("datafiles/ekp_$(mod_type).jld2")
    ekpobj = loaded["ekpobj"]
    ekp_samples = loaded["ekp_samples"]
    prior = loaded["prior_obj"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]
    true_parameter = loaded["true_parameter"]
    αs = loaded["alphas"]
    mask = [1, length(αs)] # don't do all cases
    αs = αs[mask]

    min_iter = 1
    max_iter = min(10, length(αs)) # number of EKP iterations to use data from is at most this
    training_points = [ekp_samples[α] for α in αs[min_iter:max_iter]]

    encoder_schedule_ref = [(decorrelate_structure_mat(), "in_and_out")]

    # chosen some truncations with similar dimension
    rvs_rinfos = [(0.995, 0.99999), (0.99, 0.9999), (0.98, 0.999), (0.95, 0.99), (0.9, 0.95)]
    all_errs = zeros(length(rvs_rinfos), 1 + length(αs))

    # as we have more input samples than out stored in ekp. we provide the extended set of outputs to be used in the likelihood informed processor
    final_samples_out = ekp_samples[αs[end]][2]
    encoder_kwargs = encoder_kwargs_from(ekpobj, prior; final_samples_out = final_samples_out)

    flat_io_pairs = get_training_points(ekpobj, min_iter:max_iter, g_final = final_samples_out)

    names = ["reference", "PCA-in-out", ["LI-in(1:$(i))-out(1:1)" for i in mask]...]


    if mod_type == "linear"
        ni_scaling = 0.0 # noise injection when eval forward map (def. 1)
    else
        bad_model(mod_type, mod_types)
    end

    reduced_dims = zeros(Int, length(rvs_rinfos), length(αs) + 2) # store reduced dims for final table

    for (idx, rv_rinfo) in enumerate(rvs_rinfos)
        rv, rinfo = rv_rinfo
        encoder_schedule_decorrelate = [(decorrelate_structure_mat(retain_var = rv), "in_and_out")]
        encoder_schedules_li = [
            [
                (decorrelate_structure_mat(), "in_and_out"),
                (likelihood_informed(retain_info = rinfo, iters = 1:i), "in"),
                (likelihood_informed(retain_info = rinfo, iters = 1:1), "out"),
            ] for i in mask
        ]

        em_ref = forward_map_wrapper(
            param -> forward_map(param, model),
            prior,
            flat_io_pairs;
            encoder_schedule = deepcopy(encoder_schedule_ref),
            encoder_kwargs = deepcopy(encoder_kwargs),
            noise_injector_scaling = ni_scaling, # shouldnt be needed
        )

        em_decorrelate = forward_map_wrapper(
            param -> forward_map(param, model),
            prior,
            flat_io_pairs;
            encoder_schedule = deepcopy(encoder_schedule_decorrelate),
            encoder_kwargs = deepcopy(encoder_kwargs),
            noise_injector_scaling = ni_scaling,
        )

        ems_li = [
            forward_map_wrapper(
                param -> forward_map(param, model),
                prior,
                flat_io_pairs;
                encoder_schedule = deepcopy(encoder_schedule),
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
                @info "Truncation criteria (var>$(rv), or information > $(rinfo))"
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

    df = DataFrame(all_errs, [Symbol(n) for n in names[2:end]])

    # build dataframe
    pairs = [(reduced_dims[i, j + 1], all_errs[i, j]) for i in axes(all_errs, 1), j in axes(all_errs, 2)]

    df = DataFrame(pairs, Symbol.(names[2:end]))  # columns = names
    df.truncation = rvs_rinfos            # add row labels
    return df, all_errs, reduced_dims[:, 2:end]
end

df, all_errs, red_dims = main()

@info "(reduced_dim, error) of posterior mean to whitened \"reference\" solution,\n when using the truncation ({for PCA}, {for LI})"
select!(df, :truncation, :)
