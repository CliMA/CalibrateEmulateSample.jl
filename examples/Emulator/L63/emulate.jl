using OrdinaryDiffEq
using Random, Distributions, LinearAlgebra
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

# CES 
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.DataContainers


function lorenz(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

function main()

    # rng
    rng = MersenneTwister(1232434)

    # Run L63 from 0 -> tmax
    u0 = [1.0; 0.0; 0.0]
    tmax = 20
    dt = 0.01
    tspan = (0.0, tmax)
    prob = ODEProblem(lorenz, u0, tspan)
    sol = solve(prob, Euler(), dt = dt)

    # Run L63 from end for test trajetory data
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

    # Create training pairs (with noise) from subsampling [burnin,tmax] 
    tburn = 10
    burnin = Int(floor(10 / dt))
    n_train_pts = 400 #600 
    ind = shuffle!(rng, Vector(burnin:(tmax / dt - 1)))[1:n_train_pts]
    n_tp = length(ind)
    input = zeros(3, n_tp)
    output = zeros(3, n_tp)
    Γy = 1e-6 * I(3)
    noise = rand(rng, MvNormal(zeros(3), Γy), n_tp)
    for i in 1:n_tp
        input[:, i] = sol.u[i]
        output[:, i] = sol.u[i + 1] + noise[:, i]
    end
    iopairs = PairedDataContainer(input, output)

    # Emulate
    cases = ["GP", "RF-scalar", "RF-scalar-diagin", "RF-vector-svd-nonsep"]

    case = cases[4]
    decorrelate = true
    nugget = Float64(1e-12)
    overrides = Dict(
        "verbose" => true,
        "scheduler" => DataMisfitController(terminate_at = 1e4),
        "cov_sample_multiplier" => 2.0,
        "n_iteration" => 10,
    )

    # Build ML tools
    if case == "GP"
        gppackage = Emulators.GPJL()
        pred_type = Emulators.YType()
        mlt = GaussianProcess(
            gppackage;
            kernel = nothing, # use default squared exponential kernel
            prediction_type = pred_type,
            noise_learn = false,
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
            optimizer_options = overrides,
        )
    elseif case ∈ ["RF-vector-svd-nonsep"]
        kernel_structure = NonseparableKernel(LowRankFactor(9, nugget))
        n_features = 500

        mlt = VectorRandomFeatureInterface(
            n_features,
            3,
            3,
            rng = rng,
            kernel_structure = kernel_structure,
            optimizer_options = overrides,
        )
    end

    # Emulate
    emulator = Emulator(mlt, iopairs; obs_noise_cov = Γy, decorrelate = decorrelate)
    optimize_hyperparameters!(emulator)


    # Predict with emulator
    xspan_test = 0.0:dt:tmax_test
    u_test = zeros(3, length(xspan_test))
    u_test[:, 1] = sol_test.u[1]

    for i in 1:(length(xspan_test) - 1)
        rf_mean, _ = predict(emulator, u_test[:, i:i], transform_to_real = true) # 3x1 matrix
        u_test[:, i + 1] = rf_mean
    end
    train_err = [0.0]
    for i in 1:size(input, 2)
        train_mean, _ = predict(emulator, input[:, i:i], transform_to_real = true) # 3x1
        train_err[1] += norm(train_mean - output[:, i])
    end
    println("normalized L^2 error on training data:", 1 / size(input, 2) * train_err[1])



    # plotting trace
    f = Figure(resolution = (900, 450))
    axx = Axis(f[1, 1], xlabel = "time", ylabel = "x")
    axy = Axis(f[2, 1], xlabel = "time", ylabel = "y")
    axz = Axis(f[3, 1], xlabel = "time", ylabel = "z")

    xx = 0:dt:tmax_test
    lines!(axx, xx, u_test[1, :], color = :blue)
    lines!(axy, xx, u_test[2, :], color = :blue)
    lines!(axz, xx, u_test[3, :], color = :blue)

    # run test data 
    solplot = zeros(3, length(xspan_test))
    for i in 1:length(xspan_test)
        solplot[:, i] = sol_test.u[i] #noiseless
    end
    JLD2.save("l63_testdata.jld2", "solplot", solplot, "uplot", u_test)

    lines!(axx, xspan_test, solplot[1, :], color = :orange)
    lines!(axy, xspan_test, solplot[2, :], color = :orange)
    lines!(axz, xspan_test, solplot[3, :], color = :orange)

    current_figure()
    # save
    save("l63_test.png", f, px_per_unit = 3)
    save("l63_test.pdf", f, pt_per_unit = 3)

    # plot attractor
    f3 = Figure()
    lines(f3[1, 1], u_test[1, :], u_test[3, :], color = :blue)
    lines(f3[2, 1], solplot[1, :], solplot[3, :], color = :orange)

    # save
    save("l63_attr.png", f3, px_per_unit = 3)
    save("l63_attr.pdf", f3, pt_per_unit = 3)


    # Predict with emulator for histogram
    xspan_hist = 0.0:dt:tmax_hist
    u_hist = zeros(3, length(xspan_hist))
    u_hist[:, 1] = sol_hist.u[1]  # start at end of previous sim

    for i in 1:(length(xspan_hist) - 1)
        rf_mean, _ = predict(emulator, u_hist[:, i:i], transform_to_real = true) # 3x1 matrix
        u_hist[:, i + 1] = rf_mean
    end

    solhist = zeros(3, length(xspan_hist))
    for i in 1:length(xspan_hist)
        solhist[:, i] = sol_hist.u[i] #noiseless
    end
    JLD2.save("l63_histdata.jld2", "solhist", solhist, "uhist", u_hist)

    # plotting histograms
    f2 = Figure()
    hist(f2[1, 1], u_hist[1, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
    hist(f2[1, 2], u_hist[2, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
    hist(f2[1, 3], u_hist[3, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))

    hist!(f2[1, 1], solhist[1, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
    hist!(f2[1, 2], solhist[2, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
    hist!(f2[1, 3], solhist[3, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))

    # save
    save("l63_pdf.png", f2, px_per_unit = 3)
    save("l63_pdf.pdf", f2, pt_per_unit = 3)



end

main()
