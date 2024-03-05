# # [Setup](@id sinusoid-example)

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

### Model set up
# This file deals with setting up the model and generating pseudo-observations.

# First, we load the packages we need:
using LinearAlgebra, Random
using Plots
using JLD2
using Statistics, Distributions

# Export functions for use later
export model
export G

## Setting up the model and data for our inverse problem

# Directories
# Save the output for the emulate, sample steps.
# Output figure save directory
example_directory = @__DIR__
data_save_directory = joinpath(example_directory, "output")
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end

# We define a model which generates a sinusoid given parameters ``\theta``: an
# amplitude and a vertical shift. We will estimate these parameters from data.
# The model adds a random phase shift upon evaluation.
function model(amplitude, vert_shift; rng = Random.GLOBAL_RNG)
    # Define x-axis
    dt = 0.01
    trange = 0:dt:(2 * pi + dt)
    # Set phi
    phi = 2 * pi * rand(rng)
    return amplitude * sin.(trange .+ phi) .+ vert_shift
end

# We will define a "true" amplitude and vertical shift, to generate some pseudo-observations. 

# Generate observations using "true" parameters, amplitude_true and vert_shift_true
# and user defined observation covariance matrix, Γ (start with diagonal covariance matrix,
# e.g. Γ = 0.1 * I)
function generate_obs(amplitude_true, vert_shift_true, Γ; rng = Random.GLOBAL_RNG)
    theta_true = [amplitude_true, vert_shift_true]
    # Generate the "true" signal for these parameters
    signal_true = model(amplitude_true, vert_shift_true; rng = rng)
    # Plot
    p = plot(signal_true, color = :black, linewidth = 3, label = "True signal")

    # We will observe properties of the signal that inform us about the amplitude and vertical 
    # position. These properties will be the range (the difference between the maximum and the minimum),
    # which is informative about the amplitude of the sinusoid, and the mean, which is informative 
    # about the vertical shift. 
    y1_true = maximum(signal_true) - minimum(signal_true)
    y2_true = mean(signal_true)
    # Add true observables to plot in red
    hline!(
        [y2_true],
        color = :red,
        style = :dash,
        linewidth = 2,
        label = "True mean: " * string(round(y2_true, digits = 2)),
    )
    plot!(
        [argmax(signal_true), argmax(signal_true)],
        [minimum(signal_true), maximum(signal_true)],
        arrows = :both,
        color = :red,
        linewidth = 2,
        label = "True range: " * string(round(y1_true, digits = 2)),
    )

    # However, our observations are typically not noise-free, so we add some white noise to our 
    # observables, defined by the covariance matrix Γ (usually assumed diagonal)
    dim_output = 2
    white_noise = MvNormal(zeros(dim_output), Γ)
    y_obs = [y1_true, y2_true] .+ rand(rng, white_noise)
    y1_obs = y_obs[1]
    y2_obs = y_obs[2]
    # Add noisy observables to plot in blue
    hline!(
        [y2_obs],
        color = :blue,
        style = :dash,
        linewidth = 1,
        label = "Observed mean: " * string(round(y2_obs, digits = 2)),
    )
    plot!(
        [argmax(signal_true) + 10, argmax(signal_true) + 10],
        [y2_true - y1_obs / 2, y2_true + y1_obs / 2],
        arrows = :both,
        color = :blue,
        linewidth = 1,
        label = "Observed range: " * string(round(y1_obs, digits = 2)),
    )

    savefig(p, joinpath(data_save_directory, "sinusoid_true_vs_observed_signal.png"))

    # Save our observations and the parameters that generated them
    println("Observations:", y_obs)

    save(joinpath(data_save_directory, "observations.jld2"), "y_obs", y_obs, "theta_true", theta_true, "Γ", Γ)
    return y_obs
end

# It will be helpful for us to define a function ``G(\theta)``, which returns these observables 
# (the range and the mean) of the sinusoid given a parameter vector. 
function G(theta; rng = Random.GLOBAL_RNG)
    amplitude, vert_shift = theta
    sincurve = model(amplitude, vert_shift; rng = rng)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end
