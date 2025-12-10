using JLD2, TOML
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
# Requires Julia <v1.12, EKI v2.1.0, JLD2 v0.5.11
eki_filename = "eki.jld2"
@load eki_filename eki


# get the priors
priors_filename = "priors_experiment3_0.toml"
param_dict = TOML.parsefile(priors_filename)
names = ["a_bar1", "a_bar2", "a_bar3", "a_bar4", "a_bar5", "x_1", "x_2", "x_3", "x_4", "x_5", "drs1", "drs2", "drs3", "drs4", "drs5", "l1", "l2"]
param_true_constrained=[5e5 ,5e5*1.05 ,5e5*1.2 ,5e5 ,5e5,0.1,0.75,0.666666,0.666666,0.1,8e-3*1,8e-3*1.5,8e-3*1.1,8e-3*1.8,8e-3*1,7.5e3,7.5e3] 
prior_vec = [get_parameter_distribution(param_dict, n) for n in names]
prior = combine_distributions(prior_vec)


# get the training data
N_train = 8 # trains on data 1:skip:skip*N_train
skip = 2
N_iterations_max = length(get_u(eki))
train_iterations = 1:skip:min(skip*N_train, N_iterations_max - 2)
train_pairs = get_training_points(eki, train_iterations)

test_iterations = train_iterations[end]+1:N_iterations_max-1
test_pairs = get_training_points(eki, test_iterations)

# prune NaNs (mainly in first few iterations)
nan_cols = findall( sum(isnan.(get_outputs(train_pairs)),dims=1)[:] .> 0)
not_nan_cols = setdiff(collect(1:size(get_outputs(train_pairs),2)), nan_cols)
train_pairs = PairedDataContainer(get_inputs(train_pairs)[:,not_nan_cols],get_outputs(train_pairs)[:,not_nan_cols])

nan_cols = findall( sum(isnan.(get_outputs(test_pairs)),dims=1)[:] .> 0)
not_nan_cols = setdiff(collect(1:size(get_outputs(test_pairs),2)), nan_cols)
test_pairs = PairedDataContainer(get_inputs(test_pairs)[:,not_nan_cols],get_outputs(test_pairs)[:,not_nan_cols])

                                  
# Create the ML tool (GP first)
gppackage = GPJL()
mlt = GaussianProcess(gppackage, noise_learn = false)

# data processing & dim reduction
retain_var = 0.95 # decorrelate, and reduce, retaining X fraction of variance
encoder_schedule = [(decorrelate_sample_cov(retain_var=retain_var), "in") ,(decorrelate_structure_mat(retain_var=retain_var), "out")]

# Build the emulator
emulator = Emulator(mlt, train_pairs; encoder_schedule = deepcopy(encoder_schedule), encoder_kwargs = (; obs_noise_cov = Γ), verbose=true)

# some bounds on the hyperparameters [works for GPJL()] 
n_hparams=length(get_params(mlt)[1])
low = repeat([log(1e-5)], n_hparams) # in log space
high = repeat([log(1e5)], n_hparams)

optimize_hyperparameters!(emulator; kernbounds=[low, high]) 

save("emulator.jld2", "emulator", emulator)
# Validation on test data:


# Then perform the sampling - here use pCN for 17d problem
chain_length = 100_000

init_sample = get_u_mean(eki, maximum(train_iterations))

mcmc = MCMCWrapper(pCNMHSampling(), y, prior, emulator; init_params = init_sample)
@info "Finding reasonable stepsize..."
new_step = optimize_stepsize(mcmc; rng = rng, init_stepsize = 0.1, N = 2000, discard_initial = 0)
@info "Begin MCMC - with step size $(new_step)"
chain = MarkovChainMonteCarlo.sample(mcmc, chain_length; rng = rng, stepsize = new_step, discard_initial = 2_000)

# We can print summary statistics of the MCMC chain
display(chain)

posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
save("mcmc_and_chain.jld2", "mcmc", mcmc, "chain", chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))
constrained_eki_optimal = transform_unconstrained_to_constrained(prior, init_sample)

save("posterior.jld2", "posterior", posterior, "constrained_eki_optimal", constrained_eki_optimal, "param_true_constrained", param_true_constrained)


p = plot(prior, fill = :lightgray)
plot!(posterior, fill = :darkblue, alpha = 0.5)
for (i,sp) in enumerate(p.subplots)
    vline!(sp, [constrained_eki_optimal[i]], lc="red", ls=:dash, lw=4) # eki at final iteration
    vline!(sp, [param_true_constrained[i]], lc="black", lw=4)
end
savefig(p, joinpath(@__DIR__, "posterior_marginals.png"))








