

using GlobalSensitivityAnalysis
const GSA = GlobalSensitivityAnalysis
using Distributions
using DataStructures
using Random
using LinearAlgebra

using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers

using JLD2

using CairoMakie, ColorSchemes #for plots
seed = 2589456
#=
#take in parameters x as [3 x pts] matrix
# Classical values (a,b) = (7, 0.05) from Sobol, Levitan 1999
#             also (a,b) = (7, 0.1) from Marrel et al 2009
function ishigami(x::MM; a = 7.0, b = 0.05) where {MM <: AbstractMatrix}
    @assert size(x,1) == 3
    return (1 .+ b * x[3,:].^4) * sin.(x[1,:]) + a * sin.(x[2,:]).^2 
end
=#

output_directory = joinpath(@__DIR__, "output")
if !isdir(output_directory)
    mkdir(output_directory)
end


function main()

    rng = MersenneTwister(seed)

    n_repeats = 20 # repeat exp with same data.

    # To create the sampling
    n_data_gen = 2000

    data = SobolData(
        params = OrderedDict(:x1 => Uniform(-π, π), :x2 => Uniform(-π, π), :x3 => Uniform(-π, π)),
        N = n_data_gen,
    )

    # To perform global analysis,
    # one must generate samples using Sobol sequence (i.e. creates more than N points)
    samples = GSA.sample(data)
    n_data = size(samples, 1) # [n_samples x 3]
    # run model (example)
    y = GSA.ishigami(samples)
    # perform Sobol Analysis
    result = analyze(data, y)

    f1 = Figure(resolution = (1.618 * 900, 300), markersize = 4)
    axx = Axis(f1[1, 1], xlabel = "x1", ylabel = "f")
    axy = Axis(f1[1, 2], xlabel = "x2", ylabel = "f")
    axz = Axis(f1[1, 3], xlabel = "x3", ylabel = "f")

    scatter!(axx, samples[:, 1], y[:], color = :orange)
    scatter!(axy, samples[:, 2], y[:], color = :orange)
    scatter!(axz, samples[:, 3], y[:], color = :orange)

    save(joinpath(output_directory, "ishigami_slices_truth.png"), f1, px_per_unit = 3)
    save(joinpath(output_directory, "ishigami_slices_truth.pdf"), f1, px_per_unit = 3)

    n_train_pts = 300
    ind = shuffle!(rng, Vector(1:n_data))[1:n_train_pts]
    # now subsample the samples data
    n_tp = length(ind)
    input = zeros(3, n_tp)
    output = zeros(1, n_tp)
    Γ = 1e-2
    noise = rand(rng, Normal(0, Γ), n_tp)
    for i in 1:n_tp
        input[:, i] = samples[ind[i], :]
        output[i] = y[ind[i]] + noise[i]
    end
    iopairs = PairedDataContainer(input, output)

    cases = ["Prior", "GP", "RF-scalar"]
    case = cases[3]
    decorrelate = true
    nugget = Float64(1e-12)
    overrides = Dict(
        "scheduler" => DataMisfitController(terminate_at = 1e4),
        "n_features_opt" => 150,
        "n_ensemble" => 30,
        "n_iteration" => 20,
        "accelerator" => NesterovAccelerator(),
    )
    if case == "Prior"
        # don't do anything
        overrides["n_iteration"] = 0
        overrides["cov_sample_multiplier"] = 0.1
    end

    y_preds = []
    result_preds = []
    opt_diagnostics = []
    for rep_idx in 1:n_repeats

        # Build ML tools
        if case == "GP"
            gppackage = Emulators.SKLJL()
            pred_type = Emulators.YType()
            mlt = GaussianProcess(gppackage; prediction_type = pred_type, noise_learn = false)

        elseif case ∈ ["RF-scalar", "Prior"]

            kernel_structure = SeparableKernel(LowRankFactor(3, nugget), OneDimFactor())
            n_features = 500
            mlt = ScalarRandomFeatureInterface(
                n_features,
                3,
                rng = rng,
                kernel_structure = kernel_structure,
                optimizer_options = deepcopy(overrides),
            )
        end

        # Emulate
        emulator = Emulator(mlt, iopairs; obs_noise_cov = Γ * I, decorrelate = decorrelate)
        optimize_hyperparameters!(emulator)

        # get EKP errors - just stored in "optimizer" box for now
        if case == "RF-scalar"
            diag_tmp = reduce(hcat, get_optimizer(mlt)) # (n_iteration, dim_output=1) convergence for each scalar mode as cols
            push!(opt_diagnostics, diag_tmp)
        end
        # predict on all Sobol points with emulator (example)    
        y_pred, y_var = predict(emulator, samples', transform_to_real = true)

        # obtain emulated Sobol indices
        result_pred = analyze(data, y_pred')
        push!(y_preds, y_pred)
        push!(result_preds, result_pred)

        jldsave(joinpath(output_directory, "emulator_repeat_$(rep_idx)_$(case).jld2"); emulator)
    end


    # analytic sobol indices
    a = 7
    b = 0.1
    V = a^2 / 8 + b * π^4 / 5 + b^2 * π^8 / 18 + 1 / 2
    V1 = 0.5 * (1 + b * π^4 / 5)^2
    V2 = a^2 / 8
    V3 = 0
    VT1 = 0.5 * (1 + b * π^4 / 5)^2 + 8 * b^2 * π^8 / 225
    VT2 = a^2 / 8
    VT3 = 8 * b^2 * π^8 / 225

    jldsave(
        joinpath(output_directory, "results_$case.jld2");
        sobol_pts = samples,
        train_idx = ind,
        mlt_pred_y = y_preds,
        mlt_sobol = result_preds,
        analytic_sobol = [V1, V2, V3, VT1, VT2, VT3],
        true_y = y,
        noise_y = Γ,
        estimated_sobol = result,
    )

    println(" ")
    println("True Sobol Indices")
    println("******************")
    println("    firstorder: ", [V1 / V, V2 / V, V3 / V])
    println("    totalorder: ", [VT1 / V, VT2 / V, VT3 / V])
    println(" ")
    println("Sampled truth Sobol Indices (# points $n_data)")
    println("***************************")
    println("    firstorder: ", result[:firstorder])
    println("    totalorder: ", result[:totalorder])
    println(" ")

    println("Sampled Emulated Sobol Indices (# obs $n_train_pts, noise var $Γ)")
    println("***************************************************************")
    if n_repeats == 1
        println("    firstorder: ", result_preds[1][:firstorder])
        println("    totalorder: ", result_preds[1][:totalorder])
    else
        firstorder_mean = mean([rp[:firstorder] for rp in result_preds])
        firstorder_std = std([rp[:firstorder] for rp in result_preds])
        totalorder_mean = mean([rp[:totalorder] for rp in result_preds])
        totalorder_std = std([rp[:totalorder] for rp in result_preds])

        println("(mean) firstorder: ", firstorder_mean)
        println("(std)  firstorder: ", firstorder_std)
        println("(mean) totalorder: ", totalorder_mean)
        println("(std)  totalorder: ", totalorder_std)
    end

    # plots

    f2 = Figure(resolution = (1.618 * 900, 300), markersize = 4)
    axx_em = Axis(f2[1, 1], xlabel = "x1", ylabel = "f")
    axy_em = Axis(f2[1, 2], xlabel = "x2", ylabel = "f")
    axz_em = Axis(f2[1, 3], xlabel = "x3", ylabel = "f")
    scatter!(axx_em, samples[:, 1], y_preds[1][:], color = :blue)
    scatter!(axy_em, samples[:, 2], y_preds[1][:], color = :blue)
    scatter!(axz_em, samples[:, 3], y_preds[1][:], color = :blue)
    scatter!(axx_em, samples[ind, 1], y[ind] + noise, color = :red, markersize = 8)
    scatter!(axy_em, samples[ind, 2], y[ind] + noise, color = :red, markersize = 8)
    scatter!(axz_em, samples[ind, 3], y[ind] + noise, color = :red, markersize = 8)

    save(joinpath(output_directory, "ishigami_slices_$(case).png"), f2, px_per_unit = 3)
    save(joinpath(output_directory, "ishigami_slices_$(case).pdf"), f2, px_per_unit = 3)


    if length(opt_diagnostics) > 0
        err_cols = reduce(hcat, opt_diagnostics) #error for each repeat as columns?

        #save
        error_filepath = joinpath(output_directory, "eki_conv_error.jld2")
        save(error_filepath, "error", err_cols)

        # print all repeats
        f3 = Figure(resolution = (1.618 * 300, 300), markersize = 4)
        ax_conv = Axis(f3[1, 1], xlabel = "Iteration", ylabel = "Error")

        if n_repeats == 1
            lines!(ax_conv, collect(1:size(err_cols, 1))[:], err_cols[:], solid_color = :blue) # If just one repeat
        else
            for idx in 1:size(err_cols, 1)
                err_normalized = (err_cols' ./ err_cols[1, :])' # divide each series by the max, so all errors start at 1
                series!(ax_conv, err_normalized', solid_color = :blue)
            end
        end

        save(joinpath(output_directory, "ishigami_eki-conv_$(case).png"), f3, px_per_unit = 3)
        save(joinpath(output_directory, "ishigami_eki-conv_$(case).pdf"), f3, px_per_unit = 3)

    end
end


main()
