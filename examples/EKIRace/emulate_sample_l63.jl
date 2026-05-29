# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))

# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
ENV["GKSwstype"] = "100"
using StatsPlots
using Plots
using Random
using JLD2
using Dates

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers

include("Lorenz63.jl") # Contains Lorenz 96 source code 


function main()

    method_cases= ["Inversion", "TransformInversion", "Unscented", "GaussNewtonInversion"]
    
    #### CHOOSE YOUR CASE: (todo: looped over in some fashion)
    # calib_data_dir
    method=method_cases[1]
    calibrate_date=Date("2026-05-28", "yyyy-mm-dd")
    calib_directory="$(method)_$(calibrate_date)"

    # calib_filename_suffix items to loop over
    N_enss = [10,25,40] 
    rng_idxs=collect(1:20)
    
    # emulate_sample cases
    for N_ens in N_enss
        for rng_idx in rng_idxs
            calib_filename_suffix = "$(N_ens)_$(rng_idx)"
            @info("Perform emulate-sample for L63: \n method: $(calib_directory) \n experiment: $(calib_filename_suffix)")
            min_iter = 1
            skip_iter = 1
            max_iter = 10 # number of EKP iterations to use data from is at most this 
            
            ####
            
            # need to loop over
            rng_seed = 44011
            rng = Random.MersenneTwister(rng_seed)
            
            # loading relevant data
            homedir = pwd()
            println(homedir)
            figure_save_directory = joinpath(homedir, "output/", calib_directory)
            data_save_directory = joinpath(homedir, "output", calib_directory)
            loaded_calib_files = [
                joinpath(data_save_directory, "l63_ekp_$(calib_filename_suffix).jld2"),
                joinpath(data_save_directory, "l63_priors.jld2"),
                joinpath(data_save_directory, "l63_calibrate_results_$(calib_filename_suffix).jld2"),
            ]
            if !isfile(loaded_calib_files[1])
                throw(ErrorException("data files not found. \n First run: \n > julia --project calibrate_l96.jl"))
            end
            
            ekpobj = load(loaded_calib_files[1])["ekpobj"]
            priors = load(loaded_calib_files[2])["prior"]
            truth_sample = load(loaded_calib_files[3])["y"]
            truth_params_obj = load(loaded_calib_files[3])["truth_params_structure"] #true parameters in constrained space
            truth_params_constrained = truth_params_obj.u # the parameters here have been exponentiated not using our in-built constraints.
            truth_params = transform_constrained_to_unconstrained(priors, truth_params_constrained)
            Γy = get_obs_noise_cov(ekpobj)
            
            n_params = length(truth_params) # "input dim"
            output_dim = size(Γy, 1)
            ###
            ###  Emulate: 
            ###
            # choice of machine-learning tool in the emulation stage
            nugget = 1e-6
            overrides = Dict(
                #                    "verbose" => true,
                "scheduler" => DataMisfitController(terminate_at = 1000.0),
                "cov_sample_multiplier" => 5.0,
                "n_iteration" => 10,
                "n_features_opt" => 60,
            )
            n_features = 100
            kernel_structure = SeparableKernel(DiagonalFactor(nugget), OneDimFactor())
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_params,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = overrides,
            )
            
            
            # Get training points from the EKP iteration number in the second input term  
            iter_end = length(get_u(ekpobj))-1
            N_iter = min(max_iter, iter_end) # number of paired iterations taken from EKP
            min_iter = min(max_iter, max(1, min_iter))
            if N_iter < 1
                @error("No iterations found in ekpobj, perhaps a calibration issue? Skipping case $(calib_filename_suffix)...")
                continue
            elseif N_iter == 1
                @warn("Convergence in only 1 iteration, removing train-test split based on iterations")
                @info "Train iterations: $(min_iter:skip_iter:N_iter)"
                input_output_pairs = Utilities.get_training_points(ekpobj, min_iter:skip_iter:N_iter)                    
            else
                @info "Train iterations: $(min_iter:skip_iter:(N_iter-1))"
                @info "Test iterations: $(N_iter:(length(get_u(ekpobj)) - 1))"
                input_output_pairs = Utilities.get_training_points(ekpobj, min_iter:skip_iter:(N_iter - 1))
                input_output_pairs_test = Utilities.get_training_points(ekpobj, N_iter:iter_end) #  "next" iterations
                
            end
            
            # data processing configuration
            retain_var = 0.99
            encoder_schedule = [(decorrelate_structure_mat(retain_var = retain_var), "in_and_out")]
            encoder_kwargs = encoder_kwargs_from(ekpobj, priors)
            
            emulator = Emulator(
                mlt,
                input_output_pairs;
                encoder_schedule = deepcopy(encoder_schedule),
                encoder_kwargs = encoder_kwargs,
            )
            optimize_hyperparameters!(emulator)
            
            # Check how well the Gaussian Process regression predicts on the
            # true parameters
            y_mean, y_var = Emulators.predict(emulator, reshape(truth_params, :, 1))
            if !(N_iter<2)
                y_mean_test, y_var_test = Emulators.predict(emulator, get_inputs(input_output_pairs_test))
            end
            println("ML prediction on true parameters: ")
            println(vec(y_mean))
            println("true data: ")
            println(truth_sample) # what was used as truth
            println(" ML predicted standard deviation")
            println(sqrt.(diag(y_var[1], 0)))
            println("ML weighted-MSE (truth): ")
            diff = (truth_sample - vec(y_mean))
            Γypluscov = Symmetric(mean(y_var) + Γy)
            println(mean(diff' * (Γypluscov\ diff)))
            if !(N_iter<2)
                println("ML weighted-MSE (test ensemble): ")
                difftest = get_outputs(input_output_pairs_test) - y_mean_test
                println(mean(difftest' * (Γypluscov \ difftest)))
            end
            
            #end
            ###
            ###  Sample: Markov Chain Monte Carlo
            ###
            # initial values
            u0 = vec(mean(get_inputs(input_output_pairs), dims = 2))
            println("initial parameters: ", u0)
            
            # First let's run a short chain to determine a good step size
            mcmc = MCMCWrapper(RWMHSampling(), truth_sample, priors, emulator; init_params = u0)
            new_step = optimize_stepsize(mcmc; init_stepsize = 0.2, N = 2000, discard_initial = 0)
            
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
            
            posterior_samples = reduce(vcat,[get_distribution(posterior)[name] for name in get_name(posterior)]) #samples are columns
            constrained_posterior_samples =
                mapslices(x -> transform_unconstrained_to_constrained(posterior, x), posterior_samples, dims = 1)
            
            # marginal histogram
            pp = plot(priors, c = :gray)
            plot!(pp, posterior)
            final_params_constrained = get_ϕ_mean_final(priors, ekpobj)
            for (i, sp) in enumerate(pp.subplots)
                vline!(sp, [truth_params_constrained[i]], lc = :black, lw = 4)
                vline!(sp, [final_params_constrained[i]], lc = :magenta, lw = 4)
            end
            figpath = joinpath(figure_save_directory, "posterior_hist_$(calib_filename_suffix)")
            savefig(figpath * ".png")
            savefig(figpath * ".pdf")
            
            # Save data
            save(
                joinpath(data_save_directory, "l63_posterior_$(calib_filename_suffix).jld2"),
                "posterior",
                posterior,
                "priors",
                priors,
                "truth_params_constrained",
                truth_params_constrained,
                "final_params_constrained",
                final_params_constrained,
                "input_output_pairs",
                input_output_pairs,
                "truth_params",
                truth_params,
                "encoder_schedule",
                get_encoder_schedule(emulator),
            )
        end
    end
end


main()
