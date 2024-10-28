using OrdinaryDiffEq
using Random, Distributions, LinearAlgebra
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.DataContainers
using EnsembleKalmanProcesses.Localizers

function lorenz(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

function main()

    output_directory = joinpath(@__DIR__, "output")
    if !isdir(output_directory)
        mkdir(output_directory)
    end

    # rng
    rng = MersenneTwister(1232435)

    n_repeats = 5 # repeat exp with same data.
    println("run experiment $n_repeats times")
    rank_test = 1:9 # must be 1:k for now
    n_iteration = 20

    #for later plots
    fontsize = 20
    wideticks= WilkinsonTicks(3, k_min=3, k_max=4) # prefer few ticks


    # Run L63 from 0 -> tmax
    u0 = [1.0; 0.0; 0.0]
    tmax = 20
    dt = 0.01
    tspan = (0.0, tmax)
    prob = ODEProblem(lorenz, u0, tspan)
    sol = solve(prob, Euler(), dt = dt)

    # Run L63 from end for test trajectory data
    tmax_test = 100
    tspan_test = (0.0, tmax_test)
    u0_test = sol.u[end]
    prob_test = ODEProblem(lorenz, u0_test, tspan_test)
    sol_test = solve(prob_test, Euler(), dt = dt)

    # Run L63 from end for histogram matching data
    tmax_hist = 1
    tspan_hist = (0.0, tmax_hist)
    u0_hist = sol_test.u[end]
    prob_hist = ODEProblem(lorenz, u0_hist, tspan_hist)
    sol_hist = solve(prob_hist, Euler(), dt = dt)


    # test data for plotting
    xspan_test = 0.0:dt:tmax_test
    solplot = zeros(3, length(xspan_test))
    for i in 1:length(xspan_test)
        solplot[:, i] = sol_test.u[i] #noiseless
    end

    # hist data for plotting
    xspan_hist = 0.0:dt:tmax_hist
    solhist = zeros(3, length(xspan_hist))
    for i in 1:length(xspan_hist)
        solhist[:, i] = sol_hist.u[i] #noiseless
    end

    # Create training pairs (with noise) from subsampling [burnin,tmax] 
    tburn = 1 # NB works better with no spin-up!
    burnin = Int(floor(tburn / dt))
    n_train_pts = 500
    sample_rand = true
    if sample_rand
        ind = Int.(shuffle!(rng, Vector(burnin:(tmax / dt - 1)))[1:n_train_pts])
    else
        ind = burnin:(n_train_pts + burnin)
    end
    n_tp = length(ind)
    input = zeros(3, n_tp)
    output = zeros(3, n_tp)
    noise_var = 1e-4
    Γy = noise_var * I(3)
    @info "with noise size: $(noise_var)"
    noise = rand(rng, MvNormal(zeros(3), Γy), n_tp)
    for i in 1:n_tp
        input[:, i] = sol.u[ind[i]]
        output[:, i] = sol.u[ind[i] + 1] + noise[:, i]
    end
    iopairs = PairedDataContainer(input, output)


    # Emulate
    cases = ["GP", "RF-prior", "RF-scalar", "RF-scalar-diagin", "RF-svd-nonsep", "RF-nosvd-nonsep", "RF-nosvd-sep", "RF-svd-sep"]

    case = cases[8] #5

    nugget = Float64(1e-8)
    u_test = []
    u_hist = []
    opt_diagnostics = zeros(length(rank_test),n_repeats,n_iteration)
    train_err = zeros(length(rank_test),n_repeats)
    test_err = zeros(length(rank_test),n_repeats)

    ttt = zeros(length(rank_test),n_repeats)
    for (rank_id, rank_val) in enumerate(rank_test) #test over different ranks for svd-nonsep
        @info "Test rank: $(rank_val)"
        for rep_idx in 1:n_repeats
            @info "Repeat: $(rep_idx)"
            rf_optimizer_overrides = Dict(
                "scheduler" => DataMisfitController(terminate_at = 1e4),
                "cov_sample_multiplier" => 1.0, #5.0,
                "n_features_opt" => 150,
                "n_iteration" => n_iteration,
                "accelerator" => DefaultAccelerator(),
                #"localization" => EnsembleKalmanProcesses.Localizers.SECNice(0.01,1.0), # localization / s
                "n_ensemble" => 200,
                "verbose" => true,
                "n_cross_val_sets" => 2,
            )


            # Build ML tools
            if case == "GP"
                gppackage = Emulators.GPJL()
                pred_type = Emulators.YType()
                mlt = GaussianProcess(gppackage; prediction_type = pred_type, noise_learn = false)
            elseif case ∈ ["RF-svd-nonsep"]
                kernel_structure = NonseparableKernel(LowRankFactor(rank_val, nugget))
                n_features = 500
                
                mlt = VectorRandomFeatureInterface(
                    n_features,
                    3,
                    3,
                    rng = rng,
                    kernel_structure = kernel_structure,
                    optimizer_options = rf_optimizer_overrides,
            )
            elseif case ∈ ["RF-svd-sep", "RF-nosvd-sep"]
                rank_out = Int(ceil(rank_val/3)) # 1 1 1 2 2 2 
                rank_in = rank_val - 3*(rank_out-1) # 1 2 3 1 2 3
                @info "Test rank in: $(rank_in) out: $(rank_out)"
                kernel_structure = SeparableKernel(LowRankFactor(rank_in, nugget), LowRankFactor(rank_out, nugget))
                n_features = 500
                mlt = VectorRandomFeatureInterface(
                    n_features,
                    3,
                    3,
                    rng = rng,
                    kernel_structure = kernel_structure,
                    optimizer_options = rf_optimizer_overrides,
                )

            end
            
            #save config for RF
            if !(case == "GP") && (rep_idx == 1)
                JLD2.save(
                    joinpath(output_directory, case * "_l63_config-diff-rank-test.jld2"),
                    "rf_optimizer_overrides",
                    rf_optimizer_overrides,
                    "n_features",
                    n_features,
                    "kernel_structure",
                    kernel_structure,
                )
            end

            # Emulate
            if case ∈ ["RF-nosvd-nonsep", "RF-nosvd-sep"]
                decorrelate = false
            else
                decorrelate = true
            end
            ttt[rank_id, rep_idx] = @elapsed begin
                emulator = Emulator(mlt, iopairs; obs_noise_cov = Γy, decorrelate = decorrelate)
                optimize_hyperparameters!(emulator)
            end
            
            # diagnostics
            if case != "GP"
                opt_diagnostics[rank_id,rep_idx,:] = get_optimizer(mlt)[1] #length-1 vec of vec  -> vec
            end
            
            # Predict with emulator
            u_test_tmp = zeros(3, length(xspan_test))
            u_test_tmp[:, 1] = sol_test.u[1]

            # predict sequentially i -> i+1
            for i in 1:(length(xspan_test) - 1)
                rf_mean, _ = predict(emulator, u_test_tmp[:, i:i], transform_to_real = true) # 3x1 matrix
                u_test_tmp[:, i + 1] = rf_mean
            end

            # training error i -> o
            train_err_tmp = [0.0]
            for i in 1:size(input, 2)
                train_mean, _ = predict(emulator, input[:, i:i], transform_to_real = true) # 3x1
                train_err_tmp[1] += norm(train_mean - output[:, i])
            end
            train_err[rank_id,rep_idx] = 1 / size(input, 2) * train_err_tmp[1]

            # test error i -> o
            test_err_tmp = [0.0]
            for i in 1:(length(xspan_test) - 1)
                test_mean, _ = predict(emulator, reshape(sol_test.u[i],:,1), transform_to_real = true) # 3x1 matrix
                test_err_tmp[1] += norm(test_mean[:] - sol_test.u[i+1])
            end
            test_err[rank_id,rep_idx] = 1/(length(xspan_test) - 1) * test_err_tmp[1]
            println("normalized L^2 error on training data:", 1 / size(input, 2) * train_err_tmp[1])
            
            u_hist_tmp = zeros(3, length(xspan_hist))
            u_hist_tmp[:, 1] = sol_hist.u[1]  # start at end of previous sim
            
            for i in 1:(length(xspan_hist) - 1)
                rf_mean, _ = predict(emulator, u_hist_tmp[:, i:i], transform_to_real = true) # 3x1 matrix
                u_hist_tmp[:, i + 1] = rf_mean
            end
            
            push!(u_test, u_test_tmp)
            push!(u_hist, u_hist_tmp)
            
            JLD2.save(
                 joinpath(output_directory, case * "_l63_rank_test_results.jld2"),
                     "rank_test", collect(rank_test),
                     "timings", ttt,
                     "train_err", train_err,
                     "test_err", test_err,
                 )

        end
        
    end

    # save data
    JLD2.save(joinpath(output_directory, case * "_l63_trainerr.jld2"), "train_err", train_err)
    JLD2.save(joinpath(output_directory, case * "_l63_histdata.jld2"), "solhist", solhist, "uhist", u_hist)
    JLD2.save(joinpath(output_directory, case * "_l63_testdata.jld2"), "solplot", solplot, "uplot", u_test)

    # plot eki convergence plot
    if length(opt_diagnostics) > 0
        
        err_cols = mean(opt_diagnostics, dims=2)[:,1,:]' # average error over repeats for each rank as columns
        err_normalized = (err_cols' ./ err_cols[1, :])' # divide each series by the max, so all errors start at 1
        plot_x = collect(1:size(err_cols,1))
        #save
        error_filepath = joinpath(output_directory, "eki_diff-rank-test_conv_error.jld2")
        save(error_filepath, "error", err_cols, "trajectory_error", train_err)

        f5 = Figure(resolution = (1.618 * 300, 300), markersize = 4)
        ax_conv = Axis(f5[1, 1], xlabel = "Iteration", ylabel = "max-normalized error", yscale = log10)
        for (rank_id,rank_val) in enumerate(rank_test)
                lines!(ax_conv, plot_x, err_normalized[:,rank_id], label = "rank $(rank_val)")
        end
        axislegend(ax_conv)
        save(joinpath(output_directory, "l63_diff-rank-test_eki-conv_$(case).png"), f5, px_per_unit = 3)
        save(joinpath(output_directory, "l63_diff-rank-test_eki-conv_$(case).pdf"), f5, px_per_unit = 3)

    end

   
    
    
end

main()
