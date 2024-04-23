# # [Sample](@id sinusoid-example)

# In this example we have a model that produces a sinusoid
# ``f(A, v) = A \sin(\phi + t) + v, \forall t \in [0,2\pi]``, with a random
# phase ``\phi``. We want to quantify uncertainties on parameters ``A`` and ``v``, 
# given noisy observations of the model output.
# Previously, in the emulate step, we built an emulator to allow us to make quick and 
# approximate model evaluations. This will be used in our Markov chain Monte Carlo 
# to sample the posterior distribution.

# First, we load the packages we need:
using LinearAlgebra, Random

using Distributions, Plots
using JLD2

using CalibrateEmulateSample
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo

const CES = CalibrateEmulateSample
const EKP = CalibrateEmulateSample.EnsembleKalmanProcesses

# Next, we need to load the emulator we built in the previous step (emulate.jl must be run before this script
# We will start with the Gaussian process emulator.
example_directory = @__DIR__
data_save_directory = joinpath(example_directory, "output")
# Get observations, true parameters and observation noise
obs_file = joinpath(data_save_directory, "observations.jld2")
y_obs = load(obs_file)["y_obs"]
theta_true = load(obs_file)["theta_true"]
Γ = load(obs_file)["Γ"]
# Get GP emulators and prior
emulator_file = joinpath(data_save_directory, "emulators.jld2")
emulator_gp = load(emulator_file)["emulator_gp"]
prior = load(emulator_file)["prior"]
# Get random number generator to start where we left off
rng = load(emulator_file)["rng"]

# We will also need a suitable value to initiate MCMC. To reduce burn-in, we will use the 
# final ensemble mean from EKI. 
calibrate_file = joinpath(data_save_directory, "calibrate_results.jld2")
ensemble_kalman_process = load(calibrate_file)["eki"]

## Markov chain Monte Carlo (MCMC)
# Here, we set up an MCMC sampler, using the API. The MCMC will be run in the unconstrained space, for computational
# efficiency. First, we need to find a suitable starting point, ideally one that is near the posterior distribution. 
# We start the MCMC from the final ensemble mean from EKI as this will increase the chance of acceptance near
# the start of the chain, and reduce burn-in time.
init_sample = EKP.get_u_mean_final(ensemble_kalman_process)
println("initial parameters: ", init_sample)

# Create MCMC from the wrapper: we will use a random walk Metropolis-Hastings MCMC (RWMHSampling())
# We need to provide the API with the observations (y_obs), priors (prior) and our emulator (emulator_gp). 
# The emulator is used because it is cheap to evaluate so we can generate many MCMC samples.
mcmc = MCMCWrapper(RWMHSampling(), y_obs, prior, emulator_gp; init_params = init_sample)
# First let's run a short chain to determine a good step size
new_step = optimize_stepsize(mcmc; rng = rng, init_stepsize = 0.1, N = 2000, discard_initial = 0)

# Now begin the actual MCMC
println("Begin MCMC - with step size ", new_step)     # 0.4
chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; rng = rng, stepsize = new_step, discard_initial = 2_000)

# We can print summary statistics of the MCMC chain
display(chain)

# Note that these values are provided in the unconstrained space. The vertical shift
# seems reasonable, but the amplitude is not. This is because the amplitude is constrained to be
# positive, but the MCMC is run in the unconstrained space.  We can transform to the real 
# constrained space and re-calculate these values.

# Extract posterior samples and plot
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))

constrained_amp = vec(constrained_posterior["amplitude"])
constrained_vshift = vec(constrained_posterior["vert_shift"])

println("Amplitude mean: ", mean(constrained_amp))
println("Amplitude std: ", std(constrained_amp))
println("Vertical Shift mean: ", mean(constrained_vshift))
println("Vertical Shift std: ", std(constrained_vshift))

# We can quickly plot priors and posterior using built-in capabilities
p = plot(prior, fill = :lightgray, rng = rng)
plot!(posterior, fill = :darkblue, alpha = 0.5, rng = rng, size = (800, 200))
savefig(p, joinpath(data_save_directory, "sinusoid_posterior_GP.png"))

# This shows the posterior distribution has collapsed around the true values for theta. 
# Note, these are marginal distributions but this is a multi-dimensional problem with a 
# multi-dimensional posterior. Marginal distributions do not show us how parameters co-vary,
# so we also plot the 2D posterior distribution.

# Plot 2D histogram (in constrained space)
amp_lims = (0, 6)           # manually set limits based on our priors
vshift_lims = (-6, 10)

hist2d = histogram2d(
    constrained_amp,
    constrained_vshift,
    colorbar = :false,
    xlims = amp_lims,
    ylims = vshift_lims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
)

# Let's also plot the marginal distributions along the top and the right hand
# panels. We can plot the prior and marginal posteriors as histograms.
prior_samples = sample(rng, prior, Int(1e4))
constrained_prior_samples = EKP.transform_unconstrained_to_constrained(prior, prior_samples)

tophist = histogram(
    constrained_prior_samples[1, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    legend = :false,
    lab = "Prior",
    yaxis = :false,
    xlims = amp_lims,
    xlabel = "Amplitude",
)
histogram!(
    tophist,
    constrained_amp,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = "Posterior",
)
righthist = histogram(
    constrained_prior_samples[2, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    orientation = :h,
    ylim = vshift_lims,
    xlims = (0, 1.4),
    xaxis = :false,
    legend = :false,
    lab = :false,
    ylabel = "Vertical Shift",
)

histogram!(
    righthist,
    constrained_vshift,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = :false,
    orientation = :h,
)

layout = @layout [
    tophist{0.8w, 0.2h} _
    hist2d{0.8w, 0.8h} righthist{0.2w, 0.8h}
]

plot_all = plot(
    tophist,
    hist2d,
    righthist,
    layout = layout,
    size = (600, 600),
    legend = :true,
    guidefontsize = 14,
    tickfontsize = 12,
    legendfontsize = 12,
)

savefig(plot_all, joinpath(data_save_directory, "sinusoid_MCMC_hist_GP.png"))


### MCMC Sampling using Random Features Emulator

# We could repeat the above process with the random features (RF) emulator in place of the GP
# emulator. We hope to see similar results, since our RF emulator should be a good approximation
# to the GP emulator.

emulator_random_features = load(emulator_file)["emulator_random_features"]
mcmc = MCMCWrapper(RWMHSampling(), y_obs, prior, emulator_random_features; init_params = init_sample)
new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

println("Begin MCMC - with step size ", new_step)      # 0.4
chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)

# We can print summary statistics of the MCMC chain
display(chain)

# The output of the random features MCMC is almost identical. Again, these are in the unconstrained space
# so we need to transform to the real (constrained) space and re-calculate these values.

# Extract posterior samples and plot
posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

# Back to constrained coordinates
constrained_posterior =
    Emulators.transform_unconstrained_to_constrained(prior, MarkovChainMonteCarlo.get_distribution(posterior))

constrained_amp = vec(constrained_posterior["amplitude"])
constrained_vshift = vec(constrained_posterior["vert_shift"])

println("Amplitude mean: ", mean(constrained_amp))
println("Amplitude std: ", std(constrained_amp))
println("Vertical shift mean: ", mean(constrained_vshift))
println("Vertical shift std: ", std(constrained_vshift))

# These numbers are very similar to our GP results. We can also check the posteriors look similar
# using the same plotting functions as before.
p = plot(prior, fill = :lightgray, rng = rng)
plot!(posterior, fill = :darkblue, alpha = 0.5, rng = rng, size = (800, 200))
savefig(p, joinpath(data_save_directory, "sinusoid_posterior_RF.png"))

# Plot 2D histogram (in constrained space)
# Using the same set up as before, with the same xlims, ylims.
hist2d = histogram2d(
    constrained_amp,
    constrained_vshift,
    colorbar = :false,
    xlims = amp_lims,
    ylims = vshift_lims,
    xlabel = "Amplitude",
    ylabel = "Vertical Shift",
)

# As before, we will plot the marginal distributions for both prior and posterior
# We will use the same prior samples generated for the GP histogram.
tophist = histogram(
    constrained_prior_samples[1, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    legend = :false,
    lab = "Prior",
    yaxis = :false,
    xlims = amp_lims,
    xlabel = "Amplitude",
)
histogram!(
    tophist,
    constrained_amp,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = "Posterior",
)
righthist = histogram(
    constrained_prior_samples[2, :],
    bins = 100,
    normed = true,
    fill = :lightgray,
    orientation = :h,
    ylim = vshift_lims,
    xlims = (0, 1.4),
    xaxis = :false,
    legend = :false,
    lab = :false,
    ylabel = "Vertical Shift",
)

histogram!(
    righthist,
    constrained_vshift,
    bins = 50,
    normed = true,
    fill = :darkblue,
    alpha = 0.5,
    legend = :false,
    lab = :false,
    orientation = :h,
)

layout = @layout [
    tophist{0.8w, 0.2h} _
    hist2d{0.8w, 0.8h} righthist{0.2w, 0.8h}
]

plot_all = plot(
    tophist,
    hist2d,
    righthist,
    layout = layout,
    size = (600, 600),
    legend = :true,
    guidefontsize = 14,
    tickfontsize = 12,
    legendfontsize = 12,
)

savefig(plot_all, joinpath(data_save_directory, "sinusoid_MCMC_hist_RF.png"))

# It is reassuring to see that this method is robust to the choice of emulator. The MCMC using 
# both GP and RF emulators give very similar posterior distributions. 
