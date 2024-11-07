using Random, Distributions, LinearAlgebra
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

function main()
    # filepaths
    output_directory = joinpath(@__DIR__, "output")
    cases = ["GP", "RF-prior", "RF-scalar", "RF-scalar-diagin", "RF-svd-nonsep", "RF-nosvd-nonsep", "RF-nosvd-sep"] #for paper, 1 2 & 5.
    case = cases[1]
    @info "plotting case $case"
    #for later plots
    fontsize = 20
    wideticks = WilkinsonTicks(3, k_min = 3, k_max = 4) # prefer few ticks

    # Load from saved files
    #=
        config_file = JLD2.load(joinpath(output_directory, case * "_l63_config.jld2"))
        rf_optimizer_overrides = config_file["rf_optimizer_overrides"]
        n_features = config_file["n_features"]
        kernel_structure = config_file["kernel_structure"]
        =#

    histogram_data = JLD2.load(joinpath(output_directory, case * "_l63_histdata.jld2"))
    solhist = histogram_data["solhist"]
    uhist = histogram_data["uhist"]
    trajectory_data = JLD2.load(joinpath(output_directory, case * "_l63_testdata.jld2"))
    solplot = trajectory_data["solplot"]
    uplot = trajectory_data["uplot"]

    # plotting trajectories for just first repeat
    uplot_tmp = uplot[1]
    uhist_tmp = uhist[1]

    # copied from emulate.jl
    dt = 0.01
    tmax_test = 20 #100
    xx = 0.0:dt:tmax_test

    f = Figure(size = (900, 450), fontsize = fontsize)
    axx = Axis(f[1, 1], ylabel = "x", yticks = wideticks)
    axy = Axis(f[2, 1], ylabel = "y", yticks = wideticks)
    axz = Axis(f[3, 1], xlabel = "time", ylabel = "z", yticks = [10, 30, 50])

    tt = 1:length(xx)
    lines!(axx, xx, uplot_tmp[1, tt], color = :blue)
    lines!(axy, xx, uplot_tmp[2, tt], color = :blue)
    lines!(axz, xx, uplot_tmp[3, tt], color = :blue)

    lines!(axx, xx, solplot[1, tt], color = :orange)
    lines!(axy, xx, solplot[2, tt], color = :orange)
    lines!(axz, xx, solplot[3, tt], color = :orange)

    current_figure()
    # save
    save(joinpath(output_directory, case * "_l63_test.png"), f, px_per_unit = 3)
    save(joinpath(output_directory, case * "_l63_test.pdf"), f, pt_per_unit = 3)
    @info "plotted trajectory in \n $(case)_l63_test.png, and $(case)_l63_test.pdf"

    # plot attractor
    f3 = Figure(fontsize = fontsize)
    lines(f3[1, 1], uplot_tmp[1, :], uplot_tmp[3, :], color = :blue)
    lines(f3[2, 1], solplot[1, :], solplot[3, :], color = :orange)

    # save
    save(joinpath(output_directory, case * "_l63_attr.png"), f3, px_per_unit = 3)
    save(joinpath(output_directory, case * "_l63_attr.pdf"), f3, pt_per_unit = 3)
    @info "plotted attractor in \n $(case)_l63_attr.png, and $(case)_l63_attr.pdf"

    # plotting histograms
    f2 = Figure(fontsize = 1.25 * fontsize)
    axx = Axis(f2[1, 1], xlabel = "x", ylabel = "pdf", xticks = wideticks, yticklabelsvisible = false)
    axy = Axis(f2[1, 2], xlabel = "y", xticks = wideticks, yticklabelsvisible = false)
    axz = Axis(f2[1, 3], xlabel = "z", xticks = [10, 30, 50], yticklabelsvisible = false)

    hist!(axx, uhist_tmp[1, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
    hist!(axy, uhist_tmp[2, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
    hist!(axz, uhist_tmp[3, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))

    hist!(axx, solhist[1, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
    hist!(axy, solhist[2, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
    hist!(axz, solhist[3, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))

    # save
    save(joinpath(output_directory, case * "_l63_pdf.png"), f2, px_per_unit = 3)
    save(joinpath(output_directory, case * "_l63_pdf.pdf"), f2, pt_per_unit = 3)
    @info "plotted histogram in \n $(case)_l63_pdf.png, and $(case)_l63_pdf.pdf"

    # compare  marginal histograms to truth - rough measure of fit
    solcdf = sort(solhist, dims = 2)

    if length(uhist) > 1
        ucdf = []
        for u in uhist
            ucdf_tmp = sort(u, dims = 2)
            push!(ucdf, ucdf_tmp)
        end

        f4 = Figure(size = (900, Int(floor(900 / 1.618))), fontsize = 1.5 * fontsize)
        axx = Axis(f4[1, 1], xlabel = "x", ylabel = "cdf", xticks = wideticks)
        axy = Axis(f4[1, 2], xlabel = "y", xticks = wideticks, yticklabelsvisible = false)
        axz = Axis(f4[1, 3], xlabel = "z", xticks = [10, 30, 50], yticklabelsvisible = false)


        unif_samples = (1:size(solcdf, 2)) / size(solcdf, 2)

        for u in ucdf
            lines!(axx, u[1, :], unif_samples, color = (:blue, 0.2), linewidth = 4)
            lines!(axy, u[2, :], unif_samples, color = (:blue, 0.2), linewidth = 4)
            lines!(axz, u[3, :], unif_samples, color = (:blue, 0.2), linewidth = 4)
        end

        lines!(axx, solcdf[1, :], unif_samples, color = (:orange, 1.0), linewidth = 4)
        lines!(axy, solcdf[2, :], unif_samples, color = (:orange, 1.0), linewidth = 4)
        lines!(axz, solcdf[3, :], unif_samples, color = (:orange, 1.0), linewidth = 4)

        # save
        save(joinpath(output_directory, case * "_l63_cdfs.png"), f4, px_per_unit = 3)
        save(joinpath(output_directory, case * "_l63_cdfs.pdf"), f4, pt_per_unit = 3)
        @info "plotted cdfs for all samples in \n $(case)_l63_cdfs.png, and $(case)_l63_cdfs.pdf"
    end
end

main()
