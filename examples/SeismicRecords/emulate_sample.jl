using JLD2
using Plots
using Distributions, LinearAlgebra, Statistics, Random
using CalibrateEmulateSample
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.TOMLInterface

rng = Random.MersenneTwister(60732)

# Given calibration setup in files:
exp_data_filename = "5500Experiment3.jld2"
@load exp_data_filename y Γ

# get the eki object
# Requires EKI vX.Y.Z
# Requires JLD2 vX.Y.Z
eki_filename = "eki.jld2"
@load eki_filename eki

# get the priors
priors_filename = "priors_experiment3_0.toml"
param_dict = TOML.parsefile(toml_path)
names = ["a_bar1", "a_bar2", "a_bar3", "a_bar4", "a_bar5", "x_1", "x_2", "x_3", "x_4", "x_5", "drs1", "drs2", "drs3", "drs4", "drs5", "l1", "l2"]
prior_vec = [get_parameter_distribution(param_dict, n) for n in names]
prior = combine_distributions(prior_vec)


# get the training data
N_train = 10 # trains on data 1:N_train
N_iterations_max = length(get_u(eki))
train_iterations = 1:min(N_train , N_iterations_max - 1)
train_pairs = get_training_points(eki, train_iterations)

test_iterations = train_iterations+1:N_iterations_max
test_pairs = get_training_points(eki, test_iterations)

# Create the ML tool (GP first)
gppackage = GPJL()
mlt = GaussianProcess(gppackage, noise_learn = false)

# data processing & dim reduction
retain_var = 0.99 # decorrelate, and reduce, retaining X fraction of variance
encoder_schedule = (decorrelate_structure_mat(retain_var=retain_var), "out")
encoder_kwargs = (obs_cov_noise = Γ,)

# Build the emulator
emulator = Emulator(mlt, train_pairs; encoder_schedule = encoder_schedule, enocder_kwargs = encoder_kwargs)
optimize_hyperparameters!(emulator)


# Validation on test data:


# Then perform the sampling - here use pCN for 17d problem
chain_length = 100_000

init_sample = EKP.get_u_mean_final(eki)

mcmc = MCMCWrapper(pCNMHSampling(), y, prior, emulator; init_params = init_sample)
@info "Finding reasonable stepsize..."
new_step = optimize_stepsize(mcmc; rng = rng, init_stepsize = 0.1, N = 2000, discard_initial = 0)
@info "Begin MCMC - with step size $(new_step)"
chain = MarkovChainMonteCarlo.sample(mcmc, chain_length; rng = rng, stepsize = new_step, discard_initial = 2_000)

# We can print summary statistics of the MCMC chain
display(chain)

posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))

p = plot(prior, fill = :lightgray)
plot!(posterior, fill = :darkblue, alpha = 0.5)
savefig(p, joinpath(data_save_directory, "posterior_marginals.png"))










