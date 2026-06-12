using JLD2
using Distributions
using LinearAlgebra
using Random
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.EnsembleKalmanProcesses

include("experiment_config.jl")
include("Lorenz63.jl")

const n_pushforward_samples = 1000

function pushforward_one(cfg, N_ens, rng_idx; method = method_cases[1])
    calib_dir           = calib_directory(method, cfg)
    homedir             = joinpath(pwd())
    data_save_directory = joinpath(homedir, "output", calib_dir)

    post_fn = joinpath(data_save_directory, posterior_filename(cfg, N_ens, rng_idx))

    if !isfile(post_fn)
        @warn "No posterior file for $(case_suffix(cfg, N_ens, rng_idx)); skipping."
        return
    end

    loaded_p = JLD2.load(post_fn)
    if haskey(loaded_p, "pushforward_output_samples")
        @info "Pushforward already present in $(post_fn); skipping."
        return
    end

    prelim_file = joinpath(homedir, "output", "l63_computed_preliminaries.jld2")
    isfile(prelim_file) || error("Prelim file not found at $(prelim_file). Run calibrate_l63.jl first.")
    prelim                 = JLD2.load(prelim_file)
    x0                     = prelim["x0"]
    nx                     = length(x0)
    ic_cov_sqrt            = prelim["ic_cov_sqrt"]
    lorenz_config_settings = prelim["lorenz_config_settings"]
    observation_config     = prelim["observation_config"]
    n_output               = 9  # 3 means + 3 variances + 3 covariances

    posteriors_by_k = loaded_p["posteriors_by_k"]
    k_values        = loaded_p["k_values"]
    n_k             = length(k_values)

    output_arr = Array{Float64}(undef, n_pushforward_samples, n_output, n_k)

    for (ki, k) in enumerate(k_values)
        post_dist = posteriors_by_k[k]
        push_unc  = sample(post_dist, n_pushforward_samples)
        push_con  = transform_unconstrained_to_constrained(post_dist, push_unc)
        @info "Pushforward k=$(k), N_ens=$(N_ens), rng_idx=$(rng_idx): $(n_pushforward_samples) Lorenz63 evals"
        for s in 1:n_pushforward_samples
            output_arr[s, :, ki] = lorenz_forward(
                EnsembleMemberConfig(push_con[:, s]),
                x0 .+ ic_cov_sqrt * rand(Normal(0.0, 1.0), nx, 1),
                lorenz_config_settings,
                observation_config,
            )
        end
    end

    JLD2.jldopen(post_fn, "r+") do f
        f["pushforward_output_samples"] = output_arr   # (n_samples, n_output, n_k)
        f["pushforward_k_values"]       = k_values
        f["pushforward_n_samples"]      = n_pushforward_samples
    end
    @info "Saved pushforward to $(post_fn)"
end

function main()
    cfg   = experiment_config(:l63)
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
