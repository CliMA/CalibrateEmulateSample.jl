using CairoMakie, ColorSchemes #for plots



function main()

    output_directory = "output"
    cases = ["Prior", "GP", "RF-scalar"]
    case = cases[3]
    n_dimensions = 20
    filename = joinpath(output_directory, "GFunction_$(case)_$(n_dimensions).jld2")
    legend = true

    (
        sobol_pts,
        train_idx,
        mlt_pred_y,
        mlt_sobol,
        analytic_V,
        analytic_TV,
        true_y,
        noise_y,
        observed_y,
        estimated_sobol,
    ) = load(
        filename,
        "sobol_pts",
        "train_idx",
        "mlt_pred_y",
        "mlt_sobol",
        "analytic_V",
        "analytic_TV",
        "true_y",
        "noise_y",
        "observed_y",
        "estimated_sobol",
    )
    
    n_repeats = length(mlt_sobol)
    fontsize = 24
    if n_repeats == 1
        f3 = Figure(markersize = 8, fontsize = fontsize)
        ax3 = Axis(f3[1, 1], xticks = 1:2:n_dimensions, ylabel = "Sobol Index", xlabel = "i")
        scatter!(
            ax3,
            1:n_dimensions,
            mlt_sobol[1][:firstorder];
            color = :red,
            markersize = 8,
            marker = :cross,
            label = "V-emulate",
        )
        scatter!(ax3, estimated_sobol[:firstorder], color = :red, markersize = 8, label = "V-approx")
        scatter!(ax3, analytic_V, color = :red, markersize = 12, marker = :xcross, label = "V-true")
        scatter!(
            ax3,
            1:n_dimensions,
            mlt_sobol[1][:totalorder];
            color = :blue,
            label = "TV-emulate",
            markersize = 8,
            marker = :cross,
        )
        scatter!(ax3, estimated_sobol[:totalorder], color = :blue, markersize = 8, label = "TV-approx")
        scatter!(ax3, analytic_TV, color = :blue, markersize = 12, marker = :xcross, label = "TV-true")
        if legend
            axislegend(ax3)
        end
        png_out = joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).png")
        pdf_out = joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).pdf")
        CairoMakie.save(png_out, f3, px_per_unit = 3)
        CairoMakie.save(pdf_out, f3, px_per_unit = 3)
        @info "Plotted sensitivities, case dim = $n_dimensions, \n storing plots in: \n $png_out \n $pdf_out"

    else
        # get percentiles:
        fo_mat = zeros(n_dimensions, n_repeats)
        to_mat = zeros(n_dimensions, n_repeats)

        for (idx, rp) in enumerate(mlt_sobol)
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
        f3 = Figure(markersize = 8, fontsize = fontsize)
        ax3 = Axis(f3[1, 1], xticks = 1:2:n_dimensions, ylabel = "Sobol Index", xlabel = "i")
        errorbars!(
            ax3,
            1:n_dimensions,
            firstorder_med,
            firstorder_med - firstorder_low,
            firstorder_up - firstorder_med;
            whiskerwidth = 10,
            color = :red,
            label = "V-emulate",
        )
        scatter!(ax3, estimated_sobol[:firstorder], color = :red, markersize = 8, label = "V-approx")
        scatter!(ax3, analytic_V, color = :red, markersize = 12, marker = :xcross, label = "V-true")
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
        scatter!(ax3, estimated_sobol[:totalorder], color = :blue, markersize = 8, label = "TV-approx")
        scatter!(ax3, analytic_TV, color = :blue, markersize = 12, marker = :xcross, label = "TV-true")
        if legend
            axislegend(ax3)
        end
        png_out = joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).png")
        pdf_out = joinpath(output_directory, "GFunction_sens_$(case)_$(n_dimensions).pdf")
        CairoMakie.save(png_out, f3, px_per_unit = 3)
        CairoMakie.save(pdf_out, f3, px_per_unit = 3)
        @info "Plotted sensitivities, case dim = $n_dimensions, \n storing plots in: \n $png_out \n $pdf_out"

    end
    # plots - first 3 dimensions
    plot_dim = n_dimensions >= 3 ? 3 : n_dimensions
    f2 = Figure(resolution = (1.618 * plot_dim * 300, 300), markersize = 4, fontsize = fontsize)
    for i in 1:plot_dim
        ax2 = Axis(f2[1, i], xlabel = "x" * string(i), ylabel = "f")
        scatter!(ax2, sobol_pts[:, i], mlt_pred_y[1][:], color = :blue)
        scatter!(ax2, sobol_pts[train_idx, i], observed_y[:], color = :red, markersize = 8)
    end
    png_out = joinpath(output_directory, "GFunction_slices_$(case)_$(n_dimensions).png")
    pdf_out = joinpath(output_directory, "GFunction_slices_$(case)_$(n_dimensions).pdf")
    CairoMakie.save(png_out, f2, px_per_unit = 3)
    CairoMakie.save(pdf_out, f2, px_per_unit = 3)
    @info "Plotted slices, case dim = $n_dimensions, \n storing plots in: \n $png_out \n $pdf_out"

end


main()
