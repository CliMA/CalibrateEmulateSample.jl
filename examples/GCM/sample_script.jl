# Script to run Emulation and Sampling on data from GCM

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using Plots
using Random
using JLD2
# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

rng_seed = 2413798
Random.seed!(rng_seed)

function main()

    expname = "srf_newprior"
    # Output figure save directory
    example_directory = @__DIR__
    println(example_directory)
    figure_save_directory = joinpath(example_directory, "output")
    data_save_directory = joinpath(example_directory, "output")
    if !isdir(figure_save_directory)
        mkdir(figure_save_directory)
    end
    if !isdir(data_save_directory)
        mkdir(data_save_directory)
    end

    # [1. ] Loading and processing 
    datafile = "data_from_eki_inflateyonly_100.jld2"
    inputs = load(datafile)["inputs"] #100 x 2 x 10
    outputs = load(datafile)["outputs"] #100 x 96 x 10
    truth = load(datafile)["truth"] # 96
    obs_noise_cov = load(datafile)["obs_noise_cov"] # 96 x 96

    # process the data as in emulate_script.jl
    iter_mask = 1:4
    data_mask = 1:96
    
    inputs = inputs[:, :, iter_mask]
    outputs = outputs[:, data_mask, iter_mask]
    obs_noise_cov = obs_noise_cov[data_mask, data_mask]
    truth = truth[data_mask]

    N_ens, input_dim, N_iter = size(inputs)
    output_dim = size(outputs, 2)

    stacked_inputs = reshape(permutedims(inputs, (1, 3, 2)), (N_ens * N_iter, input_dim))
    stacked_outputs = reshape(permutedims(outputs, (1, 3, 2)), (N_ens * N_iter, output_dim))
    input_output_pairs = PairedDataContainer(stacked_inputs, stacked_outputs, data_are_columns = false) #data are rows

    # Load emulator from file
    emulatorfile = joinpath(data_save_directory,"emulator_$(expname).jld2")
    emulator = load(emulatorfile)["emulator"]

    # Load the prior from file
    priorfile = "priors.jld2"
    prior = load(priorfile)["prior"]
    
    # [2. ] Initialize and adapt the sampling scheme
    # initial values
    u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
    println("initial parameters: ", u0)
    
    # First let's run a short chain to determine a good step size
    mcmc = MCMCWrapper(RWMHSampling(), truth, prior, emulator; init_params = u0)
    new_step = optimize_stepsize(mcmc; init_stepsize = 1.0, N = 2000, discard_initial = 0)
    
    # [3. ] The actual MCMC
    println("Begin MCMC - with step size ", new_step)
    chain = MarkovChainMonteCarlo.sample(mcmc, 100_000; stepsize = new_step, discard_initial = 2_000)
    posterior = MarkovChainMonteCarlo.get_posterior(mcmc, chain)
    
    post_mean = mean(posterior)
    post_cov = cov(posterior)
    println("post_mean")
    println(post_mean)
    println("post_cov")
    println(post_cov)
    println("trace and determinant of inverse posterior covariance ")
    println(tr(inv(post_cov)), " ",det(inv(post_cov)))

        constrained_truth_params = transform_unconstrained_to_constrained(posterior, truth_params)
        param_names = get_name(posterior)

        posterior_samples = vcat([get_distribution(posterior)[name] for name in get_name(posterior)]...) #samples are columns
        constrained_posterior_samples =
            mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)

        gr(dpi = 300, size = (300, 300))
        p = cornerplot(permutedims(constrained_posterior_samples, (2, 1)), label = param_names, compact = true)
        plot!(p.subplots[1], [constrained_truth_params[1]], seriestype = "vline", w = 1.5, c = :steelblue, ls = :dash) # vline on top histogram
        plot!(p.subplots[3], [constrained_truth_params[2]], seriestype = "hline", w = 1.5, c = :steelblue, ls = :dash) # hline on right histogram
        plot!(p.subplots[2], [constrained_truth_params[1]], seriestype = "vline", w = 1.5, c = :steelblue, ls = :dash) # v & h line on scatter.
        plot!(p.subplots[2], [constrained_truth_params[2]], seriestype = "hline", w = 1.5, c = :steelblue, ls = :dash)
        figpath = joinpath(figure_save_directory, "posterior_2d-" * case * ".png")
        savefig(figpath)


        # Save data
        save(
            joinpath(data_save_directory, "posterior_$(expname).jld2"),
            "posterior",
            posterior,
            "input_output_pairs",
            input_output_pairs,
            "truth_params",
            truth_params,
        )

end


@time begin
    main()
end
