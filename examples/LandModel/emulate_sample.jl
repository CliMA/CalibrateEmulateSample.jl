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

# get the ekp object
ekp_filename = "ekp.jld2"
@load ekp_filename ekp

# output file dir:
case = "experiment_1"
out_dir= joinpath(@__DIR__,"output_$(case)")
if !isdir(out_dir)
    mkdir(out_dir)
end
# copy this script into the output directory
cp(abspath(@__FILE__), joinpath(out_dir, basename(@__FILE__)); force=true)
@info "Running case $case, \n outputs will be stored in $out_dir"

# get the priors
priors_filename = "priors.toml"
param_dict = TOML.parsefile(priors_filename)
names = ["param_1","param_2","param_3"]
prior_vec = [get_parameter_distribution(param_dict, n) for n in names]
prior = combine_distributions(prior_vec)


# get the training data
N_train = 7 # trains on data 1:skip:skip*N_train
skip = 2
N_iterations_max = length(get_u(ekp))
train_iterations = 1:skip:min(skip*N_train, N_iterations_max - 2)
@info "Using train_iterations $(train_iterations)"
train_pairs = get_training_points(ekp, train_iterations)

test_iterations = setdiff(1:N_iterations_max-1, train_iterations)
test_pairs = get_training_points(ekp, test_iterations)

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

# data processing & dim reduction - To be Configured
retain_var_in = 0.99 # decorrelate, and reduce, retaining X fraction of variance
retain_var_out = 0.95
encoder_schedule = [
    (decorrelate_sample_cov(retain_var=retain_var_in), "in") ,
    (decorrelate_structure_mat(retain_var=retain_var_out), "out")
]

# Build the emulator
emulator = Emulator(mlt, train_pairs; encoder_schedule = deepcopy(encoder_schedule), encoder_kwargs = (; obs_noise_cov = Γ), verbose=true)

optimize_hyperparameters!(emulator) 

save(joinpath(out_dir,"emulator_$(train_iterations).jld2"), "emulator", emulator)

# Validation on test data:
y_mean_test, y_var_test = Emulators.predict(emulator, get_inputs(test_pairs), transform_to_real = true)
y_mean_train, y_var_train = Emulators.predict(emulator, get_inputs(train_pairs), transform_to_real = true)
 
println("ML MSE (train set): ")
println(mean((get_outputs(train_pairs) - y_mean_train) .^ 2))

println("ML MSE (test set): ")
println(mean((get_outputs(test_pairs) - y_mean_test) .^ 2))

# SOME OTHER TESTING HERE FOR THE EMULATOR? 

# Then perform the sampling
chain_length = 100_000

init_sample = get_u_mean(ekp, maximum(train_iterations))
observation_series = get_observation_series(ekp)

mcmc = MCMCWrapper(pCNMHSampling(), observation_series, prior, emulator; init_params = init_sample)
@info "Finding reasonable stepsize..."
new_step = optimize_stepsize(mcmc; rng = rng, init_stepsize = 0.1, N = 2000, discard_initial = 0)
@info "Begin MCMC - with step size $(new_step)"
chain = MarkovChainMonteCarlo.sample(mcmc, chain_length; rng = rng, stepsize = new_step, discard_initial = 2_000)

# We can print summary statistics of the MCMC chain
display(chain)

posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
save(joinpath(out_dir,"mcmc_and_chain_$(train_iterations).jld2"), "mcmc", mcmc, "chain", chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))
constrained_ekp_optimal = transform_unconstrained_to_constrained(prior, init_sample)

save(joinpath(out_dir,"posterior_$(train_iterations).jld2"), "posterior", posterior, "constrained_ekp_optimal", constrained_ekp_optimal)


p = plot(prior, fill = :lightgray)
plot!(posterior, fill = :darkblue, alpha = 0.5)
for (i,sp) in enumerate(p.subplots)
    vline!(sp, [constrained_ekp_optimal[i]], lc="red", ls=:dash, lw=4) # ekp at final iteration
end
savefig(p, joinpath(out_dir, "posterior_marginals_$(train_iterations).png"))

# SOME TESTING HERE FOR THE POSTERIOR?
