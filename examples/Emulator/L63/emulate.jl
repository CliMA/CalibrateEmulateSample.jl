using OrdinaryDiffEq
using Random, Distributions, LinearAlgebra
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Utilities

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

    n_repeats = 1 # repeat exp with same data.
    println("run experiment $n_repeats times")

    #for later plots
    fontsize = 20
    wideticks = WilkinsonTicks(3, k_min = 3, k_max = 4) # prefer few ticks


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
    tmax_hist = 1000
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
    cases = [
        "GP",
        "RF-prior",
        "RF-scalar",
        "RF-scalar-diagin",
        "RF-svd-nonsep",
        "RF-nosvd-nonsep",
        "RF-nosvd-sep",
        "RF-svd-sep",
    ]

    case = cases[3] # 7

    nugget = Float64(1e-8)
    u_test = []
    u_hist = []
    train_err = []
    opt_diagnostics = []
    for rep_idx in 1:n_repeats
        @info "Repeat: $(rep_idx)"
        rf_optimizer_overrides = Dict(
            "scheduler" => DataMisfitController(terminate_at = 1e4),
            "cov_sample_multiplier" => 1.0, #5.0,
            "n_features_opt" => 150,
            "n_iteration" => 10,
            "n_ensemble" => 200,
            "verbose" => true,
            "n_cross_val_sets" => 2,
        )


        # Build ML tools
        if case == "GP"
            gppackage = Emulators.GPJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(gppackage; prediction_type = pred_type, noise_learn = false)
        elseif case == "RF-prior"
            #No optimization
            rf_optimizer_overrides["n_iteration"] = 0
            rf_optimizer_overrides["cov_sample_multiplier"] = 0.1
            # put in whatever you want to reflect
            kernel_structure = SeparableKernel(LowRankFactor(3, nugget), LowRankFactor(3, nugget))
            n_features = 500
            mlt = VectorRandomFeatureInterface(
                n_features,
                3,
                3,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = rf_optimizer_overrides,
            )

        elseif case ∈ ["RF-scalar", "RF-scalar-diagin"]
            n_features = 10 * Int(floor(sqrt(3 * n_tp)))
            kernel_structure =
                case == "RF-scalar-diagin" ? SeparableKernel(DiagonalFactor(nugget), OneDimFactor()) :
                SeparableKernel(LowRankFactor(3, nugget), OneDimFactor())
            mlt = ScalarRandomFeatureInterface(
                n_features,
                3,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = rf_optimizer_overrides,
            )
        elseif case ∈ ["RF-svd-nonsep"]
            kernel_structure = NonseparableKernel(LowRankFactor(4, nugget))
            n_features = 500

            mlt = VectorRandomFeatureInterface(
                n_features,
                3,
                3,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = rf_optimizer_overrides,
            )
        elseif case ∈ ["RF-nosvd-nonsep"]
            kernel_structure = NonseparableKernel(LowRankFactor(4, nugget))
            n_features = 500
            mlt = VectorRandomFeatureInterface(
                n_features,
                3,
                3,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = rf_optimizer_overrides,
            )
        elseif case ∈ ["RF-nosvd-sep", "RF-svd-sep"]
            kernel_structure = SeparableKernel(LowRankFactor(3, nugget), LowRankFactor(1, nugget))
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
                joinpath(output_directory, case * "_l63_config.jld2"),
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
            encoder_schedule = [] # no encoding
        else
            encoder_schedule = (decorrelate_structure_mat(), "out")
        end
        emulator = Emulator(mlt, iopairs; output_structure_matrix = Γy, encoder_schedule = deepcopy(encoder_schedule))
        optimize_hyperparameters!(emulator)

        # diagnostics
        if case ∈ ["RF-svd-nonsep", "RF-nosvd-nonsep", "RF-svd-sep"]
            push!(opt_diagnostics, get_optimizer(mlt)[1]) #length-1 vec of vec  -> vec
        end


        # Predict with emulator
        u_test_tmp = zeros(3, length(xspan_test))
        u_test_tmp[:, 1] = sol_test.u[1]

        for i in 1:(length(xspan_test) - 1)
            rf_mean, _ = predict(emulator, u_test_tmp[:, i:i], transform_to_real = true) # 3x1 matrix
            u_test_tmp[:, i + 1] = rf_mean
        end

        train_err_tmp = [0.0]
        for i in 1:size(input, 2)
            train_mean, _ = predict(emulator, input[:, i:i], transform_to_real = true) # 3x1
            train_err_tmp[1] += norm(train_mean - output[:, i])
        end
        println("normalized L^2 error on training data:", 1 / size(input, 2) * train_err_tmp[1])

        u_hist_tmp = zeros(3, length(xspan_hist))
        u_hist_tmp[:, 1] = sol_hist.u[1]  # start at end of previous sim

        for i in 1:(length(xspan_hist) - 1)
            rf_mean, _ = predict(emulator, u_hist_tmp[:, i:i], transform_to_real = true) # 3x1 matrix
            u_hist_tmp[:, i + 1] = rf_mean
        end

        push!(train_err, train_err_tmp)
        push!(u_test, u_test_tmp)
        push!(u_hist, u_hist_tmp)

        # plots for the first repeat
        if rep_idx == 1

            # plotting trace
            f = Figure(size = (900, 450), fontsize = fontsize)
            axx = Axis(f[1, 1], ylabel = "x", yticks = wideticks)
            axy = Axis(f[2, 1], ylabel = "y", yticks = wideticks)
            axz = Axis(f[3, 1], xlabel = "time", ylabel = "z", yticks = [10, 30, 50])

            xx = 0:dt:tmax_test
            lines!(axx, xx, u_test_tmp[1, :], color = :blue)
            lines!(axy, xx, u_test_tmp[2, :], color = :blue)
            lines!(axz, xx, u_test_tmp[3, :], color = :blue)

            lines!(axx, xspan_test, solplot[1, :], color = :orange)
            lines!(axy, xspan_test, solplot[2, :], color = :orange)
            lines!(axz, xspan_test, solplot[3, :], color = :orange)

            current_figure()
            # save
            save(joinpath(output_directory, case * "_l63_test.png"), f, px_per_unit = 3)
            save(joinpath(output_directory, case * "_l63_test.pdf"), f, pt_per_unit = 3)

            # plot attractor
            f3 = Figure(fontsize = fontsize)
            lines(f3[1, 1], u_test_tmp[1, :], u_test_tmp[3, :], color = :blue)
            lines(f3[2, 1], solplot[1, :], solplot[3, :], color = :orange)

            # save
            save(joinpath(output_directory, case * "_l63_attr.png"), f3, px_per_unit = 3)
            save(joinpath(output_directory, case * "_l63_attr.pdf"), f3, pt_per_unit = 3)

            # plotting histograms
            f2 = Figure(fontsize = 1.25 * fontsize)
            axx = Axis(f2[1, 1], xlabel = "x", ylabel = "pdf", xticks = wideticks, yticklabelsvisible = false)
            axy = Axis(f2[1, 2], xlabel = "y", xticks = wideticks, yticklabelsvisible = false)
            axz = Axis(f2[1, 3], xlabel = "z", xticks = [10, 30, 50], yticklabelsvisible = false)

            hist!(axx, u_hist_tmp[1, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
            hist!(axy, u_hist_tmp[2, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
            hist!(axz, u_hist_tmp[3, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))

            hist!(axx, solhist[1, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
            hist!(axy, solhist[2, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
            hist!(axz, solhist[3, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))

            # save
            save(joinpath(output_directory, case * "_l63_pdf.png"), f2, px_per_unit = 3)
            save(joinpath(output_directory, case * "_l63_pdf.pdf"), f2, pt_per_unit = 3)
        end

    end

    # save data
    JLD2.save(joinpath(output_directory, case * "_l63_trainerr.jld2"), "train_err", train_err)
    JLD2.save(joinpath(output_directory, case * "_l63_histdata.jld2"), "solhist", solhist, "uhist", u_hist)
    JLD2.save(joinpath(output_directory, case * "_l63_testdata.jld2"), "solplot", solplot, "uplot", u_test)

    # plot eki convergence plot
    if length(opt_diagnostics) > 0
        err_cols = reduce(hcat, opt_diagnostics) #error for each repeat as columns?

        #save
        error_filepath = joinpath(output_directory, "eki_conv_error.jld2")
        save(error_filepath, "error", err_cols)

        # print all repeats
        f5 = Figure(resolution = (1.618 * 300, 300), markersize = 4)
        ax_conv = Axis(f5[1, 1], xlabel = "Iteration", ylabel = "max-normalized error", yscale = log10)
        if n_repeats == 1
            lines!(ax_conv, collect(1:size(err_cols, 1))[:], err_cols[:], color = :blue) # If just one repeat
        else
            for idx in 1:size(err_cols, 1)
                err_normalized = (err_cols' ./ err_cols[1, :])' # divide each series by the max, so all errors start at 1
                series!(ax_conv, err_normalized', solid_color = :blue)
            end
        end
        save(joinpath(output_directory, "l63_eki-conv_$(case).png"), f5, px_per_unit = 3)
        save(joinpath(output_directory, "l63_eki-conv_$(case).pdf"), f5, px_per_unit = 3)

    end

    # compare  marginal histograms to truth - rough measure of fit
    sol_cdf = sort(solhist, dims = 2)

    u_cdf = []
    for u in u_hist
        u_cdf_tmp = sort(u, dims = 2)
        push!(u_cdf, u_cdf_tmp)
    end

    f4 = Figure(size = (900, Int(floor(900 / 1.618))), fontsize = 1.5 * fontsize)
    axx = Axis(f4[1, 1], xlabel = "x", ylabel = "cdf", xticks = wideticks)
    axy = Axis(f4[1, 2], xlabel = "y", xticks = wideticks, yticklabelsvisible = false)
    axz = Axis(f4[1, 3], xlabel = "z", xticks = [10, 30, 50], yticklabelsvisible = false)

    unif_samples = (1:size(sol_cdf, 2)) / size(sol_cdf, 2)

    for u in u_cdf
        lines!(axx, u[1, :], unif_samples, color = (:blue, 0.2), linewidth = 4)
        lines!(axy, u[2, :], unif_samples, color = (:blue, 0.2), linewidth = 4)
        lines!(axz, u[3, :], unif_samples, color = (:blue, 0.2), linewidth = 4)
    end

    lines!(axx, sol_cdf[1, :], unif_samples, color = (:orange, 1.0), linewidth = 4)
    lines!(axy, sol_cdf[2, :], unif_samples, color = (:orange, 1.0), linewidth = 4)
    lines!(axz, sol_cdf[3, :], unif_samples, color = (:orange, 1.0), linewidth = 4)

    # save
    save(joinpath(output_directory, case * "_l63_cdfs.png"), f4, px_per_unit = 3)
    save(joinpath(output_directory, case * "_l63_cdfs.pdf"), f4, pt_per_unit = 3)

end

main()
