using JLD2
using Distributions
using LinearAlgebra
using Random
using Flux
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("experiment_config.jl")
include("Lorenz96.jl")

const n_pushforward_samples = 1000

function pushforward_one(cfg, N_ens, rng_idx; method = method_cases[1])
    force_case          = cfg.force_case
    calib_dir           = calib_directory(method, cfg)
    homedir             = joinpath(pwd())
    data_save_directory = joinpath(homedir, "output", calib_dir)

    post_fn  = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))
    calib_fn = joinpath(data_save_directory, results_filename(cfg, N_ens, rng_idx))

    if !isfile(post_fn)
        @warn "No posterior file for $(case_suffix(cfg, N_ens, rng_idx)); skipping."
        return
    end

    loaded_p = JLD2.load(post_fn)
    if haskey(loaded_p, "pushforward_output_samples")
        @info "Pushforward already present in $(post_fn); skipping."
        return
    end

    prelim_file = joinpath(homedir, "output", "l96_computed_preliminaries_$(force_case).jld2")
    isfile(prelim_file) || error("Prelim file not found at $(prelim_file). Run calibrate_l96.jl first.")
    prelim                 = JLD2.load(prelim_file)
    x0                     = prelim["x0"]
    nx                     = length(x0)
    ic_cov_sqrt            = prelim["ic_cov_sqrt"]
    lorenz_config_settings = prelim["lorenz_config_settings"]
    observation_config     = prelim["observation_config"]
    n_output               = 2 * nx

    calib_loaded = JLD2.load(calib_fn)
    (truth_phi, _) = calib_loaded["truth_params_structure"]
    (phi_structure, sample_range) = if force_case == "const-force"
        (nothing, nothing)
    elseif force_case == "vec-force"
        (nothing, nothing)
    elseif force_case == "flux-force"
        (truth_phi.model, truth_phi.sample_range)
    end
    n_forcing = force_case == "flux-force" ? length(sample_range) : nx

    truth_params_constrained = loaded_p["truth_params_constrained"]
    truth_emc                = build_forcing(truth_phi, truth_params_constrained, phi_structure, sample_range)
    truth_forcing_vec        = forcing(truth_emc, x0)

    posteriors_by_k = loaded_p["posteriors_by_k"]
    k_values        = loaded_p["k_values"]
    n_k             = length(k_values)

    forcing_arr = Array{Float64}(undef, n_pushforward_samples, n_forcing, n_k)
    output_arr  = Array{Float64}(undef, n_pushforward_samples, n_output,  n_k)

    for (ki, k) in enumerate(k_values)
        post_dist = posteriors_by_k[k]
        push_unc  = sample(post_dist, n_pushforward_samples)
        push_con  = transform_unconstrained_to_constrained(post_dist, push_unc)
        @info "Pushforward k=$(k), N_ens=$(N_ens), rng_idx=$(rng_idx): $(n_pushforward_samples) Lorenz96 evals"
        for s in 1:n_pushforward_samples
            emc               = build_forcing(truth_phi, push_con[:, s], phi_structure, sample_range)
            forcing_arr[s, :, ki] = forcing(emc, x0)
            output_arr[s, :, ki]  = lorenz_forward(
                emc,
                x0 .+ ic_cov_sqrt * rand(Normal(0.0, 1.0), nx, 1),
                lorenz_config_settings,
                observation_config,
            )
        end
    end

    JLD2.jldopen(post_fn, "r+") do f
        f["pushforward_forcing_samples"] = forcing_arr    # (n_samples, n_forcing, n_k)
        f["pushforward_output_samples"]  = output_arr     # (n_samples, n_output,  n_k)
        f["pushforward_k_values"]        = k_values
        f["pushforward_n_samples"]       = n_pushforward_samples
        f["truth_forcing"]               = truth_forcing_vec
    end
    @info "Saved pushforward to $(post_fn)"
end

function main()
    exp = l96_experiment()
    @assert exp in (:l96_const, :l96_vec, :l96_flux) "EXPERIMENT must be :l96_const, :l96_vec, or :l96_flux (got $exp)"
    cfg   = experiment_config(exp)
    tasks = flat_tasks(cfg)
    idx   = task_index_from_args()

    if isnothing(idx)
        for (N_ens, rng_idx) in tasks
            pushforward_one(cfg, N_ens, rng_idx)
        end
    else
        idx < 1 || idx > length(tasks) && error("Task index $(idx) out of range 1:$(length(tasks))")
        N_ens, rng_idx = tasks[idx]
        pushforward_one(cfg, N_ens, rng_idx)
    end
end

main()
