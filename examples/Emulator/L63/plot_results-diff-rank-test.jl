using Random, Distributions, LinearAlgebra, LaTeXStrings
ENV["GKSwstype"] = "100"
using CairoMakie, ColorSchemes #for plots
using JLD2

function main()
    # filepaths
    output_directory = joinpath(@__DIR__, "output")
    #    case = "RF-svd-nonsep"
    case = "RF-svd-sep"
    rank_test = 1:9 # must be 1:k for now
    @info "plotting case $case for rank test $(rank_test)"

    #for later plots
    fontsize = 18
    wideticks = WilkinsonTicks(3, k_min = 3, k_max = 4) # prefer few ticks

    error_filepath = joinpath(output_directory, "eki_diff-rank-test_conv_error.jld2")
    error_data = JLD2.load(error_filepath)
    err_cols = error_data["error"] # exps are columns

    rank_test_filepath = joinpath(output_directory, case * "_l63_rank_test_results.jld2")
    rank_test_data = JLD2.load(rank_test_filepath)
    train_err = rank_test_data["train_err"]
    test_err = rank_test_data["test_err"]

    # Plot convergences
    err_normalized = (err_cols' ./ err_cols[1, :])' # divide each series by the max, so all errors start at 1
    n_iterations = size(err_cols, 1)
    plot_x = collect(1:n_iterations)

    f5 = Figure(resolution = (1.618 * 300, 300), markersize = 4, fontsize = fontsize)
    ax_conv = Axis(
        f5[1, 1],
        xlabel = "Iteration",
        ylabel = "max-normalized error",
        yscale = log10,
        xticks = 2:2:n_iterations,
        yticks = ([10^-4, 10^-2, 10^0, 10^2], [L"10^{-4}", L"10^{-2}", L"10^0", L"10^2"]),
    )
    plot_to = 10
    for (rank_id, rank_val) in enumerate(rank_test)
        lines!(
            ax_conv,
            plot_x[1:plot_to],
            err_normalized[1:plot_to, rank_id],
            label = "rank $(rank_val)",
            color = (:blue, 0.5 * rank_val / 10),
        )
    end

    #axislegend(ax_conv)
    save(joinpath(output_directory, "l63_diff-rank-test_eki-conv_$(case).png"), f5, px_per_unit = 3)
    save(joinpath(output_directory, "l63_diff-rank-test_eki-conv_$(case).pdf"), f5, px_per_unit = 3)

    # if including GP - either load or just put in average here:
    gp_train_test_err = [0.0154, 0.00292]

    f6 = Figure(resolution = (1.618 * 300, 300), markersize = 4, fontsize = fontsize)
    if case == "RF-svd-nonsep"
        ax_error = Axis(
            f6[1, 1],
            xlabel = "rank",
            xticks = collect(rank_test),
            ylabel = "L²-error in trajectory",
            yscale = log10,
        )
        mean_train_err = mean(train_err[1:length(rank_test), :], dims = 2)
        mean_test_err = mean(test_err[1:length(rank_test), :], dims = 2)
        lines!(ax_error, rank_test, mean_train_err[:], label = "RF-train", color = :blue, linestyle = :dash)
        lines!(ax_error, rank_test, mean_test_err[:], label = "RF-test", color = :blue)
        hlines!(ax_error, [gp_train_test_err[1]], label = "GP-train", color = :orange, linestyle = :dash)
        hlines!(ax_error, [gp_train_test_err[2]], label = "GP-test", color = :orange, linestyle = :solid)

        axislegend(ax_error)
    elseif case == "RF-svd-sep"
        rank_labels = []
        for rank_val in collect(rank_test)
            rank_out = Int(ceil(rank_val / 3)) # 1 1 1 2 2 2 
            rank_in = rank_val - 3 * (rank_out - 1) # 1 2 3 1 2 3
            push!(rank_labels, (rank_in, rank_out))
        end
        idx = [1, 4, 7, 2, 5, 8, 3, 6, 9]

        ax_error = Axis(
            f6[1, 1],
            xlabel = "rank (in,out)",
            xticks = (rank_test, ["$(rank_labels[i])" for i in idx]),
            ylabel = "L²-error in trajectory",
            yscale = log10,
        )
        mean_train_err = mean(train_err[idx, :], dims = 2)
        mean_test_err = mean(test_err[idx, :], dims = 2)
        lines!(ax_error, rank_test, mean_train_err[:], label = "RF-train", color = :blue, linestyle = :dash)
        lines!(ax_error, rank_test, mean_test_err[:], label = "RF-test", color = :blue)
        hlines!(ax_error, [gp_train_test_err[1]], label = "GP-train", color = :orange, linestyle = :dash)
        hlines!(ax_error, [gp_train_test_err[2]], label = "GP-test", color = :orange, linestyle = :solid)
        axislegend(ax_error)
    else
        throw(ArgumentError("case $(case) not found, check for typos"))
    end

    save(joinpath(output_directory, "l63_diff-rank-test_train_err_$(case).png"), f6, px_per_unit = 3)
    save(joinpath(output_directory, "l63_diff-rank-test_train_err_$(case).pdf"), f6, px_per_unit = 3)
end

main()
