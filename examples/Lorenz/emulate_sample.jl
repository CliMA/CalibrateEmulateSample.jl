# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))
include(joinpath(@__DIR__, "GModel.jl")) # Contains Lorenz 96 source code

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Observations

function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2)
    return norm_factor
end

function get_standardizing_factors(data::Array{FT, 1}) where {FT}
    # Input: data size: N_data*N_ensembles (splatted)
    # Ensemble median of the data
    norm_factor = median(data)
    return norm_factor
end


function main()
    #####
    # Should be loaded:
    τc = 5.0
    F_true = 8.0 # Mean F
    A_true = 2.5 # Transient F amplitude
    ω_true = 2.0 * π / (360.0 / τc) # Frequency of the transient F
    ####

    exp_name = "Lorenz_histogram_F$(F_true)_A$(A_true)-w$(ω_true)"
    rng_seed = 44009
    Random.seed!(rng_seed)
    # loading relevant data
    homedir = pwd()
    println(homedir)
    figure_save_directory = joinpath(homedir, "output/")
    data_save_directory = joinpath(homedir, "output/")
    data_save_file = joinpath(data_save_directory, "calibrate_results.jld2")

    if !isfile(data_save_file)
        throw(
            ErrorException(
                "data file $data_save_file not found. \n First run: \n > julia --project calibrate.jl \n and store results $data_save_file",
            ),
        )
    end

    ekiobj = load(data_save_file)["eki"]
    priors = load(data_save_file)["priors"]
    truth_sample = load(data_save_file)["truth_sample"]
    truth_sample_mean = load(data_save_file)["truth_sample_mean"]
    truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
    truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)
    Γy = ekiobj.obs_noise_cov
    ###
    ###  Emulate: Gaussian Process Regression
    ###

    # Emulate-sample settings
    standardize = true
    retained_svd_frac = 0.95

    gppackage = Emulators.GPJL()
    pred_type = Emulators.YType()
    gauss_proc = GaussianProcess(
        gppackage;
        kernel = nothing, # use default squared exponential kernel
        prediction_type = pred_type,
        noise_learn = false,
    )

    # Standardize the output data
    # Use median over all data since all data are the same type
    truth_sample_norm = vcat(truth_sample...)
    norm_factor = get_standardizing_factors(truth_sample_norm)
    println(size(norm_factor))
    #norm_factor = vcat(norm_factor...)
    norm_factor = fill(norm_factor, size(truth_sample))
    println("Standardization factors")
    println(norm_factor)

    # Get training points from the EKP iteration number in the second input term  
    N_iter = 5
    input_output_pairs = Utilities.get_training_points(ekiobj, N_iter)
    # Save data
    @save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

    normalized = true
    emulator = Emulator(
        gauss_proc,
        input_output_pairs;
        obs_noise_cov = Γy,
        normalize_inputs = normalized,
        standardize_outputs = standardize,
        standardize_outputs_factors = norm_factor,
        retained_svd_frac = retained_svd_frac,
    )
    optimize_hyperparameters!(emulator)

    # Check how well the Gaussian Process regression predicts on the
    # true parameters
    #if retained_svd_frac==1.0
    y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1), transform_to_real = true)

    println("GP prediction on true parameters: ")
    println(vec(y_mean))
    println(" GP variance")
    println(diag(y_var[1], 0))
    println("true data: ")
    println(truth_sample_mean) # same, regardless of norm_factor
    println("GP MSE: ")
    println(mean((truth_sample_mean - vec(y_mean)) .^ 2))

    #end
    ###
    ###  Sample: Markov Chain Monte Carlo
    ###
    # initial values
    u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
    println("initial parameters: ", u0)

    # First let's run a short chain to determine a good step size
    truth_sample = truth_sample
    mcmc = MCMCWrapper(RWMHSampling(), truth_sample, priors, emulator; init_params = u0)
    new_step = optimize_stepsize(mcmc; init_stepsize = 0.1, N = 2000, discard_initial = 0)

    # Now begin the actual MCMC
    println("Begin MCMC - with step size ", new_step)
    chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)

    post_mean = mean(posterior)
    post_cov = cov(posterior)
    println("post_mean")
    println(post_mean)
    println("post_cov")
    println(post_cov)
    println("D util")
    println(det(inv(post_cov)))
    println(" ")


    # Plot the posteriors together with the priors and the true parameter values (in the constrained space)
    n_params = length(truth_params)
    save_directory = joinpath(figure_save_directory)

    posterior_distribution_unconstrained_dict = get_distribution(posterior) #as it is a Samples, the distribution are samples
    posterior_samples_unconstrained_dict = get_distribution(posterior) #as it is a Samples, the distribution are samples
    param_names = get_name(posterior)
    posterior_samples_unconstrained_arr = vcat([posterior_samples_unconstrained_dict[n] for n in param_names]...)
    posterior_samples = transform_unconstrained_to_constrained(priors, posterior_samples_unconstrained_arr)

    for idx in 1:n_params
        if idx == 1
            param = "F"
            xbounds = [F_true - 1.0, F_true + 1.0]
            xs = collect(xbounds[1]:((xbounds[2] - xbounds[1]) / 100):xbounds[2])
        elseif idx == 2
            param = "A"
            xbounds = [A_true - 1.0, A_true + 1.0]
            xs = collect(xbounds[1]:((xbounds[2] - xbounds[1]) / 100):xbounds[2])
        elseif idx == 3
            param = "ω"
            xs = collect((ω_true - 0.2):0.01:(ω_true + 0.2))
            xbounds = [xs[1], xs[end]]
        else
            throw("not implemented")
        end

        label = "true " * param

        histogram(posterior_samples[idx, :], bins = 100, normed = true, fill = :slategray, lab = "posterior")
        prior_plot = get_distribution(mcmc.prior)

        # This requires StatsPlots
        xs_rows = permutedims(repeat(xs, 1, size(posterior_samples, 1)), (2, 1)) # This hack will enable us to apply the transformation using the full prior
        xs_rows_unconstrained = transform_constrained_to_unconstrained(priors, xs_rows)
        plot!(
            xs,
            pdf.(prior_plot[param_names[idx]], xs_rows_unconstrained[idx, :]),
            w = 2.6,
            color = :blue,
            lab = "prior",
        )
        #plot!(xs, mcmc.prior[idx].dist, w=2.6, color=:blue, lab="prior")
        plot!([truth_params[idx]], seriestype = "vline", w = 2.6, lab = label)
        plot!(xlims = xbounds)

        title!(param)

        figpath = joinpath(figure_save_directory, "posterior_$(param)_" * exp_name * ".png")
        savefig(figpath)
        linkfig(figpath)
    end

end

main()
