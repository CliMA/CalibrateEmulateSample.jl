using Distributions
using EnsembleKalmanProcesses
using JLD2
using LinearAlgebra
using Random

include("./models.jl")

## Define custom learning-rate scheduler for better control over timepoints reached
mutable struct CheckpointScheduler <: EnsembleKalmanProcesses.LearningRateScheduler
    αs::Vector{Float64}
    scheduler

    current_index

    CheckpointScheduler(αs, scheduler) = new(αs, scheduler, 2)
end
function EnsembleKalmanProcesses.calculate_timestep!(ekp, g, Δt_new, scheduler::CheckpointScheduler)
    EnsembleKalmanProcesses.calculate_timestep!(ekp, g, Δt_new, scheduler.scheduler)
    if scheduler.current_index <= length(scheduler.αs) && get_algorithm_time(ekp)[end] > scheduler.αs[scheduler.current_index]
        get_algorithm_time(ekp)[end] = scheduler.αs[scheduler.current_index]
        scheduler.current_index += 1
    end

    nothing
end
function get_algorithm_time(ekp::EnsembleKalmanProcess) # This is not defined in older package versions; this is temporary here
    return accumulate(+, ekp.Δt)
end

rng = Random.MersenneTwister(123)
input_dim = 100
output_dim = 50
αs = [0.0, 0.25, 0.5, 0.75, 1.0]

num_trials = 1
for trial in 1:num_trials
    @info "Trial $trial"

    prior_cov, y, obs_noise_cov, model, true_parameter = linlinexp(input_dim, output_dim, rng)
    prior_obj = ParameterDistribution(
        Parameterized(MvNormal(zeros(size(prior_cov, 1)), prior_cov)), fill(no_constraint(), size(prior_cov, 1)), "linlinexp_prior",
    )

    n_ensemble = 200
    initial_ensemble = construct_initial_ensemble(rng, prior_obj, n_ensemble)
    ekp = EnsembleKalmanProcess(
        initial_ensemble,
        y,
        obs_noise_cov,
        TransformInversion();
        rng,
        scheduler = CheckpointScheduler(αs, EKSStableScheduler(2.0, 0.01)),
    )

    n_iters = 0
    while vcat([0.0], get_algorithm_time(ekp))[end] < maximum(αs)
        n_iters += 1
        G_ens = hcat([forward_map(param, model) for param in eachcol(get_ϕ_final(prior_obj, ekp))]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            throw("EKI terminated prematurely: $(terminate)! Shouldn't happen...")
        end
    end
    @info "EKP iterations: $n_iters"
    @info "Loss over iterations: $(get_error(ekp))"

    ekp_samples = Dict()
    for α in αs
        closest_iter = argmin(0:n_iters) do i
            abs(α - (i == 0 ? 0.0 : get_algorithm_time(ekp)[i]))
        end + 1
        ekp_samples[α] = (get_u(ekp, closest_iter), closest_iter == n_iters + 1 ? hcat([forward_map(u, model) for u in eachcol(get_u(ekp, closest_iter))]...) : get_g(ekp, closest_iter))
    end


    #! format: off
    save(
        "datafiles/ekp_linlinexp_$(trial).jld2",
        "ekpobj", ekp,
        "ekp_samples", ekp_samples,
        "prior_obj", prior_obj,
        "y", y,
        "obs_noise_cov", obs_noise_cov,
        "model", model,
        "true_parameter", true_parameter,
    )
    #! format: on
end
