using CairoMakie, ColorSchemes #for plots



function main()

    output_directory = "output"
    cases = ["Prior", "GP", "RF-scalar"]
    case = cases[3]
    filename = joinpath(output_directory, "results_$case.jld2")

    (sobol_pts, train_idx, mlt_pred_y, mlt_sobol, analytic_sobol, true_y, noise_y, estimated_sobol) = load(
        filename,
        "sobol_pts",
        "train_idx",
        "mlt_pred_y",
        "mlt_sobol",
        "analytic_sobol",
        "true_y",
        "noise_y",
        "estimated_sobol",
    )

    n_repeats = length(mlt_sobol)
    (V, V1, V2, V3, VT1, VT2, VT3) = analytic_sobol

    f1 = Figure(resolution = (1.618 * 900, 300), markersize = 4)
    axx = Axis(f1[1, 1], xlabel = "x1", ylabel = "f")
    axy = Axis(f1[1, 2], xlabel = "x2", ylabel = "f")
    axz = Axis(f1[1, 3], xlabel = "x3", ylabel = "f")

    scatter!(axx, sobol_pts[:, 1], true_y[:], color = :orange)
    scatter!(axy, sobol_pts[:, 2], true_y[:], color = :orange)
    scatter!(axz, sobol_pts[:, 3], true_y[:], color = :orange)

    save(joinpath(output_directory, "ishigami_slices_truth.png"), f1, px_per_unit = 3)
    save(joinpath(output_directory, "ishigami_slices_truth.pdf"), f1, px_per_unit = 3)


    # display some info
    println(" ")
    println("True Sobol Indices")
    println("******************")
    println("    firstorder: ", [V1 / V, V2 / V, V3 / V])
    println("    totalorder: ", [VT1 / V, VT2 / V, VT3 / V])
    println(" ")
    println("Sampled truth Sobol Indices (# points $n_data)")
    println("***************************")
    println("    firstorder: ", estimated_sobol[:firstorder])
    println("    totalorder: ", estimated_sobol[:totalorder])
    println(" ")

    println("Sampled Emulated Sobol Indices (# obs $n_train_pts, noise var $noise_y)")
    println("***************************************************************")
    if n_repeats == 1
        println("    firstorder: ", mlt_sobol[1][:firstorder])
        println("    totalorder: ", mlt_sobol[1][:totalorder])
    else
        firstorder_mean = mean([ms[:firstorder] for ms in mlt_sobol])
        firstorder_std = std([ms[:firstorder] for ms in mlt_sobol])
        totalorder_mean = mean([ms[:totalorder] for ms in mlt_sobol])
        totalorder_std = std([ms[:totalorder] for ms in mlt_sobol])

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
    scatter!(axx_em, sobol_pts[:, 1], mlt_pred_y[1][:], color = :blue)
    scatter!(axy_em, sobol_pts[:, 2], mlt_pred_y[1][:], color = :blue)
    scatter!(axz_em, sobol_pts[:, 3], mlt_pred_y[1][:], color = :blue)
    scatter!(axx_em, sobol_pts[train_idx, 1], true_y[train_idx] + noise, color = :red, markersize = 8)
    scatter!(axy_em, sobol_pts[train_idx, 2], true_y[train_idx] + noise, color = :red, markersize = 8)
    scatter!(axz_em, sobol_pts[train_idx, 3], true_y[train_idx] + noise, color = :red, markersize = 8)

    save(joinpath(output_directory, "ishigami_slices_$(case).png"), f2, px_per_unit = 3)
    save(joinpath(output_directory, "ishigami_slices_$(case).pdf"), f2, px_per_unit = 3)




end


main()
