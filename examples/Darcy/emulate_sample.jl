# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using Plots
using Random
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

function main()

    cases = [
        "GP", # diagonalize, train scalar GP, assume diag inputs
    ]

    #### CHOOSE YOUR CASE: 
    mask = [1] # 1:8 # e.g. 1:8 or [7]
    for (case) in cases[mask]


        println("case: ", case)
        min_iter = 1
        max_iter = 5 # number of EKP iterations to use data from is at most this

        exp_name = "darcy"
        rng_seed = 940284
        rng = Random.MersenneTwister(rng_seed)

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
        prior = load(data_save_file)["prior"]
        truth_sample = load(data_save_file)["truth_sample"]
        truth_params_constrained = load(data_save_file)["truth_input_constrained"] #true parameters in constrained space
        truth_params = load(data_save_file)["truth_input_unconstrained"] #true parameters in unconstrained space  
        Γy = get_obs_noise_cov(ekiobj)


        n_params = length(truth_params) # "input dim"
        output_dim = size(Γy, 1)
        ###
        ###  Emulate: Gaussian Process Regression
        ###

        # Emulate-sample settings
        # choice of machine-learning tool in the emulation stage
        if case == "GP"
            #            gppackage = Emulators.SKLJL()
            gppackage = Emulators.GPJL()
            mlt = GaussianProcess(gppackage; noise_learn = false)
        end


        # Get training points from the EKP iteration number in the second input term  
        N_iter = min(max_iter, length(get_u(ekiobj)) - 1) # number of paired iterations taken from EKP
        min_iter = min(max_iter, max(1, min_iter))
        input_output_pairs = Utilities.get_training_points(ekiobj, min_iter:(N_iter - 1))
        input_output_pairs_test = Utilities.get_training_points(ekiobj, N_iter:(length(get_u(ekiobj)) - 1)) #  "next" iterations
        # Save data
        @save joinpath(data_save_directory, "input_output_pairs.jld2") input_output_pairs

        retained_svd_frac = 1.0
        normalized = true
        # do we want to use SVD to decorrelate outputs
        decorrelate = case ∈ ["RF-vector-nosvd-diag", "RF-vector-nosvd-nondiag"] ? false : true

        emulator = Emulator(
            mlt,
            input_output_pairs;
            obs_noise_cov = Γy,
            normalize_inputs = normalized,
            retained_svd_frac = retained_svd_frac,
            decorrelate = decorrelate,
        )
        optimize_hyperparameters!(emulator)

        # Check how well the Gaussian Process regression predicts on the
        # true parameters
        #if retained_svd_frac==1.0
        y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1), transform_to_real = true)
        y_mean_test, y_var_test =
            Emulators.predict(emulator, get_inputs(input_output_pairs_test), transform_to_real = true)

        println("ML prediction on true parameters: ")
        println(vec(y_mean))
        println("true data: ")
        println(truth_sample) # what was used as truth
        println(" ML predicted standard deviation")
        println(sqrt.(diag(y_var[1], 0)))
        println("ML MSE (truth): ")
        println(mean((truth_sample - vec(y_mean)) .^ 2))
        println("ML MSE (next ensemble): ")
        println(mean((get_outputs(input_output_pairs_test) - y_mean_test) .^ 2))

        #end
        ###
        ###  Sample: Markov Chain Monte Carlo
        ###
        # initial values
        u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
        println("initial parameters: ", u0)

        # First let's run a short chain to determine a good step size
        mcmc = MCMCWrapper(RWMHSampling(), truth_sample, prior, emulator; init_params = u0)
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

        param_names = get_name(posterior)

        posterior_samples = reduce(vcat, [get_distribution(posterior)[name] for name in get_name(posterior)]) #samples are columns of this matrix
        n_post = size(posterior_samples, 2)
        plot_sample_id = (n_post - 1000):n_post
        constrained_posterior_samples =
            transform_unconstrained_to_constrained(prior, posterior_samples[:, plot_sample_id])

        N, L = 80, 1.0
        pts_per_dim = LinRange(0, L, N)

        κ_ens_mean = reshape(mean(constrained_posterior_samples, dims = 2), N, N)
        p1 = contour(
            pts_per_dim,
            pts_per_dim,
            κ_ens_mean',
            fill = true,
            levels = 15,
            title = "kappa mean",
            colorbar = true,
        )
        κ_ens_ptw_var = reshape(var(constrained_posterior_samples, dims = 2), N, N)
        p2 = contour(
            pts_per_dim,
            pts_per_dim,
            κ_ens_ptw_var',
            fill = true,
            levels = 15,
            title = "kappa var",
            colorbar = true,
        )
        l = @layout [a b]
        plt = plot(p1, p2, layout = l)
        savefig(plt, joinpath(data_save_directory, "posterior_pointwise_uq.png"))

        # Save data
        save(
            joinpath(data_save_directory, "posterior.jld2"),
            "posterior",
            posterior,
            "input_output_pairs",
            input_output_pairs,
            "truth_params",
            truth_params,
        )
    end
end

main()
