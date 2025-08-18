# Reference the in-tree version of CalibrateEmulateSample on Julias load path
include(joinpath(@__DIR__, "..", "..", "ci", "linkfig.jl"))

# Import modules
using Random
using StableRNGs
using Distributions
using Statistics
using LinearAlgebra
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
plot_flag = true
if plot_flag
    ENV["GKSwstype"] = "100"
    using Plots
    gr(size = (1500, 700))
    font = Plots.font("Helvetica", 18)
    fontdict = Dict(:guidefont => font, :xtickfont => font, :ytickfont => font, :legendfont => font)

end


function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where {T}
    m, n = length(vy), length(vx)
    gx = reshape(repeat(vx, inner = m, outer = 1), m, n)
    gy = reshape(repeat(vy, inner = 1, outer = n), m, n)

    return gx, gy
end

function main()

    rng = MersenneTwister(41)
    output_directory = joinpath(@__DIR__, "output")
    if !isdir(output_directory)
        mkdir(output_directory)
    end

    cases = [
        "gp-skljl",
        "gp-gpjl", # Very slow prediction...
        "rf-lr-scalar",
        "rf-lr-lr",
        "rf-nonsep",
    ]

    encoder_names = ["no-proc", "sample-struct-proc", "sample-proc", "struct-mat-proc", "combined-proc", "cca-proc", "minmax"]
    encoders = [
        [], # no proc
        nothing, # default proc
        (decorrelate_sample_cov(), "in_and_out"),
        (decorrelate_structure_mat(), "in_and_out"),
        (decorrelate(), "in_and_out"),
        (canonical_correlation(), "in_and_out"),
        (minmax_scale(), "in_and_out"),
    ]

    # USER CHOICES 
    encoder_mask = [1,7] # best performing
    case_mask = [3]#[1, 3, 4, 5] # (KEEP set to 1:length(cases) when pushing for buildkite)


    # scales the problem to different in-out domains (but g gives the same "function" on all domain sizes
    out_scaling = 1e-2/(40) # scales range(g1) = 1
    in_scaling = 1/(4*pi) # scales range(x) = 1
    
    #problem
    n = 200  # number of training points
    p = 2   # input dim 
    d = 2   # output dim
    prior_cov = (pi^2) * I(p) * in_scaling * in_scaling
    X = rand(rng, MvNormal(zeros(p), prior_cov), n)

    @info "problem scaling " in_scaling out_scaling
    # G(x1, x2)
    g1(x) = out_scaling*10*(sin.(1/in_scaling*x[1, :]) .+ cos.(1/in_scaling* 2 * x[2, :]))
    g2(x) = out_scaling*10*(2 * sin.(1/in_scaling*2 * x[1, :]) .- 3 * cos.(1/in_scaling*x[2, :]))
    g1(x, y) = out_scaling*10*(sin(1/in_scaling*x) + 2 * cos(1/in_scaling*2 * y))
    g2(x, y) = out_scaling*10*(2 * sin(1/in_scaling*2 * x) - 3 * cos(1/in_scaling*y))
    g1x = g1(X)
    g2x = g2(X)
    gx = zeros(2, n)
    gx[1, :] = g1x
    gx[2, :] = g2x

    # Add noise η
    μ = zeros(d)
    #Σ = 0.1 * [[0.4, -0.3] [-0.3, 0.5]] # d x d
    #Σ = 0.001 * [[0.4, -0.0] [-0.0, 0.5]] # d x d
    Σ = out_scaling*out_scaling*1.0*I
    noise_samples = rand(rng, MvNormal(μ, Σ), n)
    # y = G(x) + η
    Y = gx .+ noise_samples

    iopairs = PairedDataContainer(X, Y, data_are_columns = true)
    @assert get_inputs(iopairs) == X
    @assert get_outputs(iopairs) == Y

    #plot training data with and without noise
    if plot_flag
        n_pts = 200
        x1 = range(-2 * pi * in_scaling, stop = 2 * π * in_scaling, length = n_pts)
        x2 = range(-2 * pi * in_scaling, stop = 2 * π * in_scaling, length = n_pts)

        p1 = contourf(x1, x2, g1.(x1', x2), c = :cividis, xlabel = "x1", ylabel = "x2", aspect_ratio = :equal)
        scatter!(p1, X[1, :], X[2, :], c = :black, ms = 3, label = "train point")

        figpath = joinpath(output_directory, "g1_true.png")
        savefig(figpath)
        p2 = contourf(x1, x2, g2.(x1', x2), c = :cividis, xlabel = "x1", ylabel = "x2", aspect_ratio = :equal)
        scatter!(p2, X[1, :], X[2, :], c = :black, ms = 3, label = "train point")

        figpath = joinpath(output_directory, "g2_true.png")
        savefig(figpath)

    end

    for case in cases[case_mask]
        for (enc_name, enc) in zip(encoder_names[encoder_mask], encoders[encoder_mask])

            println(" ")
            @info """

            -------------
            running case $case with encoder $enc_name
            -------------
            """

            # common Gaussian feature setup
            pred_type = YType()

            # common random feature setup #large n_features, small n_features_opt
            n_features = 500
            optimizer_options =
                Dict("n_iteration" => 5, "n_features_opt" => 100, "n_ensemble" => 100, "cov_sample_multiplier" => 10.0, "verbose" => true, "cov_correction" => "nice")
            nugget = 1e-12

            # data processing schedule

            if case == "gp-skljl"
                gppackage = SKLJL()
                mlt = GaussianProcess(gppackage, noise_learn = true)

            elseif case == "gp-gpjl"
                @warn "gp-gpjl case is very slow at prediction"
                gppackage = GPJL()
                mlt = GaussianProcess(gppackage, noise_learn = true)

            elseif case == "rf-lr-scalar"
                mlt = ScalarRandomFeatureInterface(
                    n_features,
                    p,
                #    kernel_structure = SeparableKernel(LowRankFactor(2, nugget), OneDimFactor()),
                    kernel_structure = SeparableKernel(DiagonalFactor(nugget), OneDimFactor()),
                    optimizer_options = optimizer_options,
                )
            elseif case == "rf-lr-lr"
                mlt = VectorRandomFeatureInterface(
                    n_features,
                    p,
                    d,
                    kernel_structure = SeparableKernel(LowRankFactor(2, nugget), LowRankFactor(2, nugget)),
                    optimizer_options = deepcopy(optimizer_options),
                )
            elseif case == "rf-nonsep"
                mlt = VectorRandomFeatureInterface(
                    n_features,
                    p,
                    d,
                    kernel_structure = NonseparableKernel(LowRankFactor(4, nugget)),
                    optimizer_options = deepcopy(optimizer_options),
                )

            end

            emulator = Emulator(
                mlt,
                iopairs,
                encoder_schedule = deepcopy(enc), # must copy! 
                input_structure_matrix = prior_cov,
                output_structure_matrix = Σ,
            )


            optimize_hyperparameters!(emulator) # although RF already optimized

            # Plot mean and variance of the predicted observables y1 and y2
            # For this, we generate test points on a x1-x2 grid.
            n_pts = 200
            x1 = range(-2 * π * in_scaling, stop = 2 * π * in_scaling, length = n_pts)
            x2 = range(-2 * π * in_scaling, stop = 2 * π * in_scaling, length = n_pts)
            X1, X2 = meshgrid(x1, x2)

            # Input for predict has to be of size  input_dim x N_samples
            inputs = permutedims(hcat(X1[:], X2[:]), (2, 1))

            em_mean, em_cov = predict(emulator, inputs, transform_to_real = true)
            println("end predictions at ", n_pts * n_pts, " points")

            g1_true = g1(inputs)
            g1_true_grid = reshape(g1_true, n_pts, n_pts)
            g2_true = g2(inputs)
            g2_true_grid = reshape(g2_true, n_pts, n_pts)
            g_true_grids = [g1_true_grid, g2_true_grid]

            #plot predictions
            for y_i in 1:d

                em_var_temp = [diag(em_cov[j]) for j in 1:length(em_cov)] # (40000,)
                em_var = permutedims(reduce(vcat, [x' for x in em_var_temp]), (2, 1)) # 2 x 40000

                mean_grid = reshape(em_mean[y_i, :], n_pts, n_pts) # 2 x 40000
                if plot_flag
                    p5 = contourf(
                        x1,
                        x2,
                        mean_grid,
                        c = :cividis,
                        xlabel = "x1",
                        ylabel = "x2",
                        title = "mean output $y_i",
                    )
                end
                var_grid = reshape(em_var[y_i, :], n_pts, n_pts)
                if plot_flag
                    p6 = contourf(
                        x1,
                        x2,
                        var_grid,
                        c = :cividis,
                        xlabel = "x1",
                        ylabel = "x2",
                        title = "var output $y_i",
                    )

                    plot(p5, p6, layout = (1, 2), legend = false)

                    savefig(joinpath(output_directory, case * "_" * enc_name * "_y" * string(y_i) * "_predictions.png"))
                end
            end

            # Plot the true components of G(x1, x2)
            g1_true = g1(inputs)
            g1_true_grid = reshape(g1_true, n_pts, n_pts)


            g2_true = g2(inputs)
            g2_true_grid = reshape(g2_true, n_pts, n_pts)
            g_true_grids = [g1_true_grid, g2_true_grid]
            MSE = 1 / size(em_mean, 2) * sqrt(sum((em_mean[1, :] - g1_true) .^ 2 + (em_mean[2, :] - g2_true) .^ 2))
            println("L^2 error of mean and latent truth:", MSE)
            #=
            # Plot the difference between the truth and the mean of the predictions
            for y_i in 1:d

            # Reshape em_cov to size N_samples x output_dim
            em_var_temp = [diag(em_cov[j]) for j in 1:length(em_cov)] # (40000,)
            em_var = permutedims(vcat([x' for x in em_var_temp]...), (2, 1)) # 40000 x 2

            mean_grid = reshape(em_mean[y_i, :], n_pts, n_pts)
            var_grid = reshape(em_var[y_i, :], n_pts, n_pts)
            # Compute and plot 1/variance * (truth - prediction)^2

            if plot_flag
                zlabel = "1/var * (true_y" * string(y_i) * " - predicted_y" * string(y_i) * ")^2"

                p9 = contourf(
                    x1,
                    x2,
                    sqrt.(1.0 ./ var_grid .* (g_true_grids[y_i] .- mean_grid) .^ 2),
                    c = :magma,
                    zlabel = zlabel,
                    xlabel = "x1",
                    ylabel = "x2",
                    title = "weighted difference $y_i"
                )

                savefig(joinpath(output_directory, case*"_"* enc_name * "_y" * string(y_i) * "_difference_truth_prediction.png"))
            end
            end
            =#
        end
    end
end

main()
