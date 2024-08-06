
using GlobalSensitivityAnalysis
const GSA = GlobalSensitivityAnalysis
using Distributions
using DataStructures
using Random
using LinearAlgebra
import StatsBase: percentile
using JLD2

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers

using CairoMakie, ColorSchemes #for plots
seed = 2589436

output_directory = joinpath(@__DIR__, "output")
if !isdir(output_directory)
    mkdir(output_directory)
end


inner_func(x::AV, a::AV) where {AV <: AbstractVector} = prod((abs.(4 * x .- 2) + a) ./ (1 .+ a))

"G-Function taken from https://www.sfu.ca/~ssurjano/gfunc.html"
function GFunction(x::AM, a::AV) where {AM <: AbstractMatrix, AV <: AbstractVector}
    @assert size(x, 1) == length(a)
    return mapslices(y -> inner_func(y, a), x; dims = 1) #applys the map to columns
end

function GFunction(x::AM) where {AM <: AbstractMatrix}
    a = [(i - 1.0) / 2.0 for i in 1:size(x, 1)]
    return GFunction(x, a)
end

function main()

    rng = MersenneTwister(seed)

    n_repeats = 3 # repeat exp with same data.
    n_dimensions = 3
    # To create the sampling
    n_data_gen = 800

    data =
        SobolData(params = OrderedDict([Pair(Symbol("x", i), Uniform(0, 1)) for i in 1:n_dimensions]), N = n_data_gen)

    # To perform global analysis,
    # one must generate samples using Sobol sequence (i.e. creates more than N points)
    samples = GSA.sample(data)
    n_data = size(samples, 1) # [n_samples x n_dim]
    println("number of sobol points: ", n_data)
    # run model (example)
    y = GFunction(samples')' # G is applied to columns
    # perform Sobol Analysis
    result = analyze(data, y)

    # plot the first 3 dimensions
    plot_dim = n_dimensions >= 3 ? 3 : n_dimensions
    f1 = Figure(resolution = (1.618 * plot_dim * 300, 300), markersize = 4)
    for i in 1:plot_dim
        ax = Axis(f1[1, i], xlabel = "x" * string(i), ylabel = "f")
        scatter!(ax, samples[:, i], y[:], color = :orange)
    end

    CairoMakie.save(joinpath(output_directory, "GFunction_slices_truth_$(n_dimensions).png"), f1, px_per_unit = 3)
    CairoMakie.save(joinpath(output_directory, "GFunction_slices_truth_$(n_dimensions).pdf"), f1, px_per_unit = 3)

    n_train_pts = n_dimensions * 250
    ind = shuffle!(rng, Vector(1:n_data))[1:n_train_pts]
    # now subsample the samples data
    n_tp = length(ind)
    input = zeros(n_dimensions, n_tp)
    output = zeros(1, n_tp)
    Γ = 1e-3
    noise = rand(rng, Normal(0, Γ), n_tp)
    for i in 1:n_tp
        input[:, i] = samples[ind[i], :]
        output[i] = y[ind[i]] + noise[i]
    end
    iopairs = PairedDataContainer(input, output)

    # analytic sobol indices taken from
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8989694/pdf/main.pdf
    a = [(i - 1.0) / 2.0 for i in 1:n_dimensions]  # a_i < a_j => a_i more sensitive
    prod_tmp = prod(1 .+ 1 ./ (3 .* (1 .+ a) .^ 2)) - 1
    V = [(1 / (3 * (1 + ai)^2)) / prod_tmp for ai in a]
    prod_tmp2 = [prod(1 .+ 1 ./ (3 .* (1 .+ a[1:end .!== j]) .^ 2)) for j in 1:n_dimensions]
    TV = [(1 / (3 * (1 + ai)^2)) * prod_tmp2[i] / prod_tmp for (i, ai) in enumerate(a)]



    cases = ["Prior", "GP", "RF-scalar"]
    case = cases[3]
    decorrelate = true
    nugget = Float64(1e-12)

    overrides = Dict(
        "verbose" => true,
        "scheduler" => DataMisfitController(terminate_at = 1e2),
        "n_features_opt" => 300,
        "train_fraction" => 0.8,#0.7
        "n_iteration" => 20,
        "cov_sample_multiplier" => 1.0,
        #        "localization" => SEC(0.1),#,Doesnt help much tbh
        #        "accelerator" => NesterovAccelerator(),
        "n_ensemble" => 200, #40*n_dimensions,
    )
    if case == "Prior"
        # don't do anything
        overrides["n_iteration"] = 0
        overrides["cov_sample_multiplier"] = 0.1
    end

    y_preds = []
    result_preds = []

    for rep_idx in 1:n_repeats
        @info "Repeat: $(rep_idx)"
        # Build ML tools
        if case == "GP"
            gppackage = Emulators.SKLJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(gppackage; prediction_type = pred_type, noise_learn = false)

        elseif case ∈ ["RF-scalar", "Prior"]
            rank = n_dimensions #<= 10 ? n_dimensions : 10
            kernel_structure = SeparableKernel(LowRankFactor(rank, nugget), OneDimFactor())
            n_features = n_dimensions <= 10 ? n_dimensions * 100 : 1000
            if (n_features / n_train_pts > 0.9) && (n_features / n_train_pts < 1.1)
                @warn "The number of features similar to the number of training points, poor performance expected, change one or other of these"
            end
            mlt = ScalarRandomFeatureInterface(
                n_features,
                n_dimensions,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = deepcopy(overrides),
            )
        end

        # Emulate
        emulator = Emulator(mlt, iopairs; obs_noise_cov = Γ * I, decorrelate = decorrelate)
        optimize_hyperparameters!(emulator)

        # predict on all Sobol points with emulator (example)    
        y_pred, y_var = predict(emulator, samples', transform_to_real = true)

        # obtain emulated Sobol indices
        result_pred = analyze(data, y_pred')
        println("First order: ", result_pred[:firstorder])
        println("Total order: ", result_pred[:totalorder])

        push!(y_preds, y_pred)
        push!(result_preds, result_pred)
        GC.gc() #collect garbage

        # PLotting:
        if rep_idx == 1
            f3, ax3, plt3 = scatter(
                1:n_dimensions,
                result_preds[1][:firstorder];
                color = :red,
                markersize = 8,
                marker = :cross,
                label = "V-emulate",
                title = "input dimension: $(n_dimensions)",
            )
            scatter!(ax3, result[:firstorder], color = :red, markersize = 8, label = "V-approx")
            scatter!(ax3, V, color = :red, markersize = 12, marker = :xcross, label = "V-true")
            scatter!(
                ax3,
                1:n_dimensions,
                result_preds[1][:totalorder];
                color = :blue,
                label = "TV-emulate",
                markersize = 8,
                marker = :cross,
            )
            scatter!(ax3, result[:totalorder], color = :blue, markersize = 8, label = "TV-approx")
            scatter!(ax3, TV, color = :blue, markersize = 12, marker = :xcross, label = "TV-true")
            axislegend(ax3)

            CairoMakie.save(
                joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).png"),
                f3,
                px_per_unit = 3,
            )
            CairoMakie.save(
                joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).pdf"),
                f3,
                px_per_unit = 3,
            )
        else
            # get percentiles:
            fo_mat = zeros(n_dimensions, rep_idx)
            to_mat = zeros(n_dimensions, rep_idx)

            for (idx, rp) in enumerate(result_preds)
                fo_mat[:, idx] = rp[:firstorder]
                to_mat[:, idx] = rp[:totalorder]
            end

            firstorder_med = percentile.(eachrow(fo_mat), 50)
            firstorder_low = percentile.(eachrow(fo_mat), 5)
            firstorder_up = percentile.(eachrow(fo_mat), 95)

            totalorder_med = percentile.(eachrow(to_mat), 50)
            totalorder_low = percentile.(eachrow(to_mat), 5)
            totalorder_up = percentile.(eachrow(to_mat), 95)

            println("(50%) firstorder: ", firstorder_med)
            println("(5%)  firstorder: ", firstorder_low)
            println("(95%)  firstorder: ", firstorder_up)

            println("(50%) totalorder: ", totalorder_med)
            println("(5%)  totalorder: ", totalorder_low)
            println("(95%)  totalorder: ", totalorder_up)
            #
            f3, ax3, plt3 = errorbars(
                1:n_dimensions,
                firstorder_med,
                firstorder_med - firstorder_low,
                firstorder_up - firstorder_med;
                whiskerwidth = 10,
                color = :red,
                label = "V-emulate",
                title = "input dimension: $(n_dimensions)",
            )
            scatter!(ax3, result[:firstorder], color = :red, markersize = 8, label = "V-approx")
            scatter!(ax3, V, color = :red, markersize = 12, marker = :xcross, label = "V-true")
            errorbars!(
                ax3,
                1:n_dimensions,
                totalorder_med,
                totalorder_med - totalorder_low,
                totalorder_up - totalorder_med;
                whiskerwidth = 10,
                color = :blue,
                label = "TV-emulate",
            )
            scatter!(ax3, result[:totalorder], color = :blue, markersize = 8, label = "TV-approx")
            scatter!(ax3, TV, color = :blue, markersize = 12, marker = :xcross, label = "TV-true")
            axislegend(ax3)

            CairoMakie.save(
                joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).png"),
                f3,
                px_per_unit = 3,
            )
            CairoMakie.save(
                joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).pdf"),
                f3,
                px_per_unit = 3,
            )

        end
        # plots - first 3 dimensions
        if rep_idx == 1
            f2 = Figure(resolution = (1.618 * plot_dim * 300, 300), markersize = 4)
            for i in 1:plot_dim
                ax2 = Axis(f2[1, i], xlabel = "x" * string(i), ylabel = "f")
                scatter!(ax2, samples[:, i], y_preds[1][:], color = :blue)
                scatter!(ax2, samples[ind, i], y[ind] + noise, color = :red, markersize = 8)
            end
            CairoMakie.save(
                joinpath(output_directory, "GFunction_slices_$(case)_$(n_dimensions).png"),
                f2,
                px_per_unit = 3,
            )
            CairoMakie.save(
                joinpath(output_directory, "GFunction_slices_$(case)_$(n_dimensions).pdf"),
                f2,
                px_per_unit = 3,
            )
        end
    end

    println(" ")
    println("True Sobol Indices")
    println("******************")
    println("    firstorder: ", V)
    println("    totalorder: ", TV)
    println(" ")
    println("Sampled truth Sobol Indices (# points $n_data)")
    println("***************************")
    println("    firstorder: ", result[:firstorder])
    println("    totalorder: ", result[:totalorder])
    println(" ")

    println("Sampled Emulated Sobol Indices (# obs $n_train_pts, noise var $Γ)")
    println("***************************************************************")


    jldsave(
        joinpath(output_directory, "Gfunction_$(case)_$(n_dimensions).jld2");
        sobol_pts = samples,
        train_idx = ind,
        analytic_V = V,
        analytic_TV = TV,
        estimated_sobol = result,
        mlt_sobol = result_preds,
        mlt_pred_y = y_preds,
        true_y = y,
        noise_y = Γ,
        observed_y = output,
    )


    return y_preds, result_preds
end


main()
