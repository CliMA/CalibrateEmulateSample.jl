# # [Calibrate](@id sinusoid-example)

################################################################################################
################################################################################################
####                Sinusoid example                                                        ####
#### This is a simple example of how to use the CalibrateEmulateSample.jl package.          ####
#### We use a simple model for a sinusoid with 2 input parameters and 2 observables.        ####
#### We follow the steps (1) Calibrate, using Ensemble Kalman Inversion, (2) Emulate, using ####
#### both Gaussian process and random features and (3) Sample, using Markov chain Monte     ####
#### Carlo. Each of these steps are run in separate scripts. For this example to work, we   ####
#### should run the scripts in the following order:                                         ####
#### (1) calibrate.jl                                                                       ####
#### (2) emulate.jl                                                                         ####
#### (3) sample.jl                                                                          ####
####                                                                                        ####
####                                                                                        ####
#### Sinusoid Model                                                                         ####
#### This example closely follows the sinusoidal model in EnsembleKalmanProcess.jl          ####
#### https://clima.github.io/EnsembleKalmanProcesses.jl/stable/literated/sinusoid_example/  ####
#### In this example we have a model that produces a sinusoid                               ####
#### ``f(A, v) = A \sin(\phi + t) + v, \forall t \in [0,2\pi]``, with a random              ####
#### phase ``\phi``. Given an initial guess of the parameters as                            ####
#### ``A^* \sim \mathcal{N}(2,1)`` and ``v^* \sim \mathcal{N}(0,25)``, our goal is          ####
#### to estimate the parameters from a noisy observation of the maximum, minimum,           ####
#### and mean of the true model output.                                                     ####
####                                                                                        ####
####                                                                                        ####
################################################################################################
################################################################################################


# First, we load the packages we need:
using LinearAlgebra, Random

using Distributions, Plots
using JLD2

# CES
using CalibrateEmulateSample

const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses
const PD = EKP.ParameterDistributions


## Setting up the model and data for our inverse problem
include("sinusoid_setup.jl")        # This file defines the model G(θ) 

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

# Generate observations (see generate_obs in sinusoid_set.jl for more details)
amplitude_true = 3.0
vert_shift_true = 7.0
theta_true = [amplitude_true, vert_shift_true]
Γ = 0.2 * I        # Observational covariance matrix (assumed diagonal)
y_obs = generate_obs(amplitude_true, vert_shift_true, Γ; rng = rng)

# Directories
# Save the output for the emulate, sample steps.
# Output figure save directory
example_directory = @__DIR__
data_save_directory = joinpath(example_directory, "output")


## Solving the inverse problem

# We now define prior distributions on the two parameters. For the amplitude,
# we define a prior with mean 2 and standard deviation 1. It is
# additionally constrained to be nonnegative. For the vertical shift we define
# a Gaussian prior with mean 0 and standard deviation 5.
prior_u1 = PD.constrained_gaussian("amplitude", 2, 1, 0, Inf)
prior_u2 = PD.constrained_gaussian("vert_shift", 0, 5, -Inf, Inf)
prior = PD.combine_distributions([prior_u1, prior_u2])
# Plot priors
p = plot(prior, fill = :lightgray, rng = rng)
savefig(p, joinpath(data_save_directory, "sinusoid_prior.png"))

# We now generate the initial ensemble and set up the ensemble Kalman inversion.
N_ensemble = 10
N_iterations = 5

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)

ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y_obs, Γ, EKP.Inversion(); rng = rng)


# We are now ready to carry out the inversion. At each iteration, we get the
# ensemble from the last iteration, apply ``G(\theta)`` to each ensemble member,
# and apply the Kalman update to the ensemble.
for i in 1:N_iterations
    params_i = EKP.get_ϕ_final(prior, ensemble_kalman_process)

    G_ens = hcat([G(params_i[:, i]; rng = rng) for i in 1:N_ensemble]...)

    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end


# Finally, we get the ensemble after the last iteration. This provides our estimate of the parameters.
final_ensemble = EKP.get_ϕ_final(prior, ensemble_kalman_process)

# Check that the ensemble mean is close to the theta_true
println("Ensemble mean: ", mean(final_ensemble, dims = 2))   # [3.08, 6.37]
println("True parameters: ", theta_true)   #[3.0, 7.0]

# Plot true model parameters and all ensemble  members at each iteration
p = plot(
    [theta_true[1]],
    [theta_true[2]],
    c = :red,
    seriestype = :scatter,
    marker = :star,
    markersize = :10,
    label = "Truth",
    legend = :bottomright,
    xlims = (0, 6),
    ylims = (-6, 10),
    guidefontsize = 14,
    tickfontsize = 12,
    legendfontsize = 12,
)
vline!([theta_true[1]], color = :red, style = :dash, label = :false)
hline!([theta_true[2]], color = :red, style = :dash, label = :false)


iteration_colormap = colormap("Blues", N_iterations)
for j in 1:N_iterations
    ensemble_j = EKP.get_ϕ(prior, ensemble_kalman_process, j)
    plot!(
        ensemble_j[1, :],
        ensemble_j[2, :],
        c = iteration_colormap[j],
        seriestype = :scatter,
        label = "Ensemble $(j)",
        alpha = 0.8,
    )
end

xlabel!("Amplitude")
ylabel!("Vertical shift")
savefig(p, joinpath(data_save_directory, "sinusoid_eki_pairs.png"))

# The ensembles are initially spread out but move closer to the true parameter values
# with each iteration, indicating the EKI algorithm is converging towards the minimum. 
# In the next step of CES, we will build an emulator using this dataset.


# Save the data
ϕ_stored = EKP.get_ϕ(prior, ensemble_kalman_process)
u_stored = EKP.get_u(ensemble_kalman_process, return_array = false)
g_stored = EKP.get_g(ensemble_kalman_process, return_array = false)

save(
    joinpath(data_save_directory, "calibrate_results.jld2"),
    "inputs",
    u_stored,
    "outputs",
    g_stored,
    "constrained_inputs",
    ϕ_stored,
    "prior",
    prior,
    "eki",
    ensemble_kalman_process,
    "N_ensemble",
    N_ensemble,
    "N_iterations",
    N_iterations,
    "rng",
    rng,
)
