using Random, Distributions, LinearAlgebra, LaTeXStrings
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

function main()
    # filepaths
    output_directory = joinpath(@__DIR__, "output")
    case = "RF-svd-nonsep"
    rank_test = 1:9 # must be 1:k for now
    @info "plotting case $case for rank test $(rank_test)"

    #for later plots
    fontsize = 20
    wideticks= WilkinsonTicks(3, k_min=3, k_max=4) # prefer few ticks

    error_filepath = joinpath(output_directory, "eki_diff-rank-test_conv_error.jld2")
    error_data = JLD2.load(error_filepath)
    err_cols =  error_data["error"] # exps are columns
    train_err = error_data["trajectory_error"]

    # Plot convergences
    
    err_normalized = (err_cols' ./ err_cols[1, :])' # divide each series by the max, so all errors start at 1
    n_iterations = size(err_cols,1)
    plot_x = collect(1:n_iterations)
    fontsize = 20

    f5 = Figure(resolution = (1.618 * 300, 300), markersize = 4, fontsize=fontsize)
    ax_conv = Axis(f5[1, 1], xlabel = "Iteration", ylabel = "max-normalized error", yscale = log10, xticks = 2:2:n_iterations, yticks= ([10^-4,10^-2,10^0, 10^2], [L"10^{-4}",L"10^{-2}", L"10^0",L"10^2"]))
    plot_to = 10
    for (rank_id,rank_val) in enumerate(rank_test)
        lines!(ax_conv, plot_x[1:plot_to], err_normalized[1:plot_to,rank_id], label = "rank $(rank_val)", color=(:blue, 0.5*rank_val/10))
    end
    #axislegend(ax_conv)
    save(joinpath(output_directory, "l63_diff-rank-test_eki-conv_$(case).png"), f5, px_per_unit = 3)
    save(joinpath(output_directory, "l63_diff-rank-test_eki-conv_$(case).pdf"), f5, px_per_unit = 3) 

    f6 = Figure(resolution = (1.618 * 300, 300), markersize = 4, fontsize=fontsize)
    ax_error = Axis(f6[1, 1], xlabel = "rank", xticks=collect(rank_test), ylabel = "L²-error in trajectory", yscale=log10 )
    mean_train_err = mean(train_err,dims=2)
    lines!(ax_error, rank_test, mean_train_err[:])
       
    save(joinpath(output_directory, "l63_diff-rank-test_train_err_$(case).png"), f6, px_per_unit = 3)
    save(joinpath(output_directory, "l63_diff-rank-test_train_err_$(case).pdf"), f6, px_per_unit = 3)
end

main()
