using Random, Distributions, LinearAlgebra
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

function main()
    # filepaths
    output_directory = joinpath(@__DIR__, "output")
    cases = ["GP", "RF-prior", "RF-scalar", "RF-scalar-diagin", "RF-svd-nonsep", "RF-nosvd-nonsep", "RF-nosvd-sep"]
    case = cases[2]

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


    f = Figure(size = (900, 450))
    axx = Axis(f[1, 1], xlabel = "time", ylabel = "x")
    axy = Axis(f[2, 1], xlabel = "time", ylabel = "y")
    axz = Axis(f[3, 1], xlabel = "time", ylabel = "z")

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

    # plot attractor
    f3 = Figure()
    lines(f3[1, 1], uplot_tmp[1, :], uplot_tmp[3, :], color = :blue)
    lines(f3[2, 1], solplot[1, :], solplot[3, :], color = :orange)

    # save
    save(joinpath(output_directory, case * "_l63_attr.png"), f3, px_per_unit = 3)
    save(joinpath(output_directory, case * "_l63_attr.pdf"), f3, pt_per_unit = 3)

    # plotting histograms
    f2 = Figure()
    hist(f2[1, 1], uhist_tmp[1, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
    hist(f2[1, 2], uhist_tmp[2, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))
    hist(f2[1, 3], uhist_tmp[3, :], bins = 50, normalization = :pdf, color = (:blue, 0.5))

    hist!(f2[1, 1], solhist[1, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
    hist!(f2[1, 2], solhist[2, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))
    hist!(f2[1, 3], solhist[3, :], bins = 50, normalization = :pdf, color = (:orange, 0.5))

    # save
    save(joinpath(output_directory, case * "_l63_pdf.png"), f2, px_per_unit = 3)
    save(joinpath(output_directory, case * "_l63_pdf.pdf"), f2, pt_per_unit = 3)



    # compare  marginal histograms to truth - rough measure of fit
    solcdf = sort(solhist, dims = 2)

    ucdf = []
    for u in uhist
        ucdf_tmp = sort(u, dims = 2)
        push!(ucdf, ucdf_tmp)
    end

    f4 = Figure(size = (900, Int(floor(900 / 1.618))))
    axx = Axis(f4[1, 1], xlabel = "", ylabel = "x")
    axy = Axis(f4[1, 2], xlabel = "", ylabel = "y")
    axz = Axis(f4[1, 3], xlabel = "", ylabel = "z")

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
end

main()
