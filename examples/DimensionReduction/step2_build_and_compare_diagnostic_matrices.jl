using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Statistics
using Distributions
using Plots
using JLD2
using Manopt, Manifolds

include("./settings.jl")
include("./problems/problem_linear.jl")
include("./problems/problem_linear_exp.jl")
include("./problems/problem_lorenz.jl")
include("./problems/problem_linlinexp.jl")

if !isfile("datafiles/ekp_$(problem)_1.jld2")
    include("step1_generate_inverse_problem_data.jl")
end

#Utilities
function cossim(x::VV1, y::VV2) where {VV1 <: AbstractVector, VV2 <: AbstractVector}
    return dot(x, y) / (norm(x) * norm(y))
end
function cossim_pos(x::VV1, y::VV2) where {VV1 <: AbstractVector, VV2 <: AbstractVector}
    return abs(cossim(x, y))
end
function cossim_cols(X::AM1, Y::AM2) where {AM1 <: AbstractMatrix, AM2 <: AbstractMatrix}
    return [cossim_pos(c1, c2) for (c1, c2) in zip(eachcol(X), eachcol(Y))]
end

all_diagnostic_matrices_u = Dict()
all_diagnostic_matrices_g = Dict()

for trial in 1:num_trials
    @info "Trial $trial"
    diagnostic_matrices_u = Dict()
    diagnostic_matrices_g = Dict()

    # Load the EKP iterations
    loaded = load("datafiles/ekp_$(problem)_$(trial).jld2")
    ekp = loaded["ekp"]
    ekp_samp = loaded["ekp_samples"]
    mcmc_samp = loaded["mcmc_samples"]
    prior = loaded["prior"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]

    prior_cov = cov(prior)
    prior_invrt = sqrt(inv(prior_cov))
    prior_rt = sqrt(prior_cov)
    obs_invrt = sqrt(inv(obs_noise_cov))
    obs_inv = inv(obs_noise_cov)

    ekp_samp_grad = Dict()
    mcmc_samp_grad = Dict()
    ekp_samp_Vgrad = Dict()
    mcmc_samp_Vgrad = Dict()
    for (dict_samp, dict_samp_grad, dict_samp_Vgrad) in (
        (ekp_samp, ekp_samp_grad, ekp_samp_Vgrad),
        (mcmc_samp, mcmc_samp_grad, mcmc_samp_Vgrad),
    )
        for (α, (samps, gsamps)) in dict_samp
            dict_samp_grad[α] = []
            for grad_type in grad_types
                @info "Computing gradients: α=$α, grad_type=$grad_type"
                grads = if grad_type == :perfect
                    jac_forward_map(samps, model)
                elseif grad_type == :mean
                    grad = jac_forward_map(reshape(mean(samps; dims = 2), :, 1), model)[1]
                    fill(grad, size(samps, 2))
                elseif grad_type == :linreg
                    grad = (gsamps .- mean(gsamps; dims = 2)) / (samps .- mean(samps; dims = 2))
                    fill(grad, size(samps, 2))
                elseif grad_type == :localsl
                    map(zip(eachcol(samps), eachcol(gsamps))) do (u, g)
                        weights = exp.(-1/2 * norm.(eachcol(u .- samps)).^2) # TODO: Matrix weighting
                        D = Diagonal(sqrt.(weights))
                        uw = (samps .- mean(samps * Diagonal(weights); dims = 2)) * D
                        gw = (gsamps .- mean(gsamps * Diagonal(weights); dims = 2)) * D
                        gw / uw
                    end
                else
                    throw("Unknown grad_type=$grad_type")
                end

                push!(dict_samp_grad[α], (grad_type, grads))
            end

            dict_samp_Vgrad[α] = []
            for Vgrad_type in Vgrad_types
                @info "Computing Vgradients: α=$α, Vgrad_type=$Vgrad_type"
                grads = if Vgrad_type == :egi
                    ∇Vs = map(enumerate(zip(eachcol(samps), eachcol(gsamps)))) do (i, (u, g))
                        @info "In full EGI procedure; particle $i/$(size(samps, 2))"

                        yys = eachcol(rand(MvNormal((1-α)g + α*y, (1-α)obs_noise_cov), α == 1.0 ? 1 : step2_Vgrad_num_samples)) # If α == 1.0, all samples will be the same anyway
                        map(yys) do yy
                            Vs = [1/2 * (yy - g)' * obs_inv * (yy - g) for g in eachcol(gsamps)]

                            X = samps[:, [1:i-1; i+1:end]] .- u
                            Z = X ./ norm.(eachcol(X))'
                            A = hcat(X'Z, (X'Z).^2 / 2)

                            ξ, γ = step2_egi_ξ, step2_egi_γ
                            Γ = γ * (factorial(3) \ Diagonal(norm.(eachcol(X)).^3) + ξ * I) # The paper has γ², but that's wrong

                            Y = Vs[[1:i-1; i+1:end]] .- Vs[i]

                            ū = pinv(Γ \ A) * (Γ \ Y)
                            Z * ū[1:end÷2]
                        end
                    end

                    mean(
                        mean(
                            ∇V * ∇V'
                            for ∇V in ∇Vs_at_x
                        ) for ∇Vs_at_x in ∇Vs
                    )
                else
                    throw("Unknown Vgrad_type=$Vgrad_type")
                end

                push!(dict_samp_Vgrad[α], (Vgrad_type, grads))
            end
        end
    end

    # random samples
    @assert 0.0 in αs
    prior_samp, prior_gsamp = ekp_samp[0.0]
    num_prior_samps = size(prior_samp, 2)

    @info "Construct PCA matrices"
    pca_u = prior_samp'
    pca_g = prior_gsamp'

    diagnostic_matrices_u["pca_u"] = pca_u, :gray
    diagnostic_matrices_g["pca_g"] = pca_g, :gray

    for α in αs
        for (sampler, dict_samp, dict_samp_grad, dict_samp_Vgrad) in (
            ("ekp", ekp_samp, ekp_samp_grad, ekp_samp_Vgrad),
            ("mcmc", mcmc_samp, mcmc_samp_grad, mcmc_samp_Vgrad),
        )
            samp, gsamp = dict_samp[α]
            for (Vgrad_type, grads) in dict_samp_Vgrad[α]
                name_suffix = "$(α)_$(sampler)_$(Vgrad_type)"
                @info "Construct $name_suffix matrices"

                Hu = prior_rt * mean(grads) * prior_rt
                # TODO: Hg

                diagnostic_matrices_u["Hu_$name_suffix"] = Hu, :black
            end

            for (grad_type, grads) in dict_samp_grad[α]
                name_suffix = "$(α)_$(sampler)_$(grad_type)"
                @info "Construct $name_suffix matrices"

                Hu = prior_rt * mean(
                    grad' * obs_inv * (
                        (1-α)obs_noise_cov + α^2 * (y - g) * (y - g)'
                    ) * obs_inv * grad
                    for (g, grad) in zip(eachcol(gsamp), grads)
                ) * prior_rt

                Hg = if α == 0
                    obs_invrt * mean(grad * prior_cov * grad' for grad in grads) * obs_invrt
                else
                    Vs0 = qr(randn(output_dim, output_dim)).Q[:, 1:step2_manopt_num_dims]

                    f = (_, Vs) -> begin
                        res = mean(
                            begin
                                mat = obs_invrt * grad * prior_rt - Vs*(Vs'*obs_invrt * grad * prior_rt)
                                a = obs_invrt * (y - g)

                                (1-α)norm(mat) + α^2 * norm(a' * mat)
                            end
                            for (g, grad) in zip(eachcol(gsamp), grads)
                        )
                        println(res)
                        res
                    end

                    egrad = Vs -> begin
                        -2mean(
                            begin
                                a = obs_invrt * (y - g)
                                mat = obs_invrt * grad * prior_cov * grad' * obs_invrt * (I - Vs * Vs') * ((1-α)I + α^2 * a * a')

                                mat + mat'
                            end
                            for (g, grad) in zip(eachcol(gsamp), grads)
                        ) * Vs
                    end
                    rgrad = (_, Vs) -> begin
                        egrd = egrad(Vs)
                        res = egrd - Vs * (Vs' * egrd)
                        res
                    end
                    Vs = quasi_Newton(Grassmann(output_dim, step2_manopt_num_dims), f, rgrad, Vs0; stopping_criterion = StopWhenGradientNormLess(3.0))
                    Vs = hcat(Vs, randn(output_dim, output_dim - step2_manopt_num_dims))
                    Vs * diagm(vcat(step2_manopt_num_dims:-1:1, zeros(output_dim - step2_manopt_num_dims))) * Vs'
                end

                diagnostic_matrices_u["Hu_$name_suffix"] = Hu, :black
                diagnostic_matrices_g["Hg_$name_suffix"] = Hg, :black
            end
        end
    end

    for (name, (value, color)) in diagnostic_matrices_u
        if !haskey(all_diagnostic_matrices_u, name)
            all_diagnostic_matrices_u[name] = ([], color)
        end
        push!(all_diagnostic_matrices_u[name][1], value)
    end
    for (name, (value, color)) in diagnostic_matrices_g
        if !haskey(all_diagnostic_matrices_g, name)
            all_diagnostic_matrices_g[name] = ([], color)
        end
        push!(all_diagnostic_matrices_g[name][1], value)
    end

    save(
        "datafiles/diagnostic_matrices_$(problem)_$(trial).jld2",
        vcat([[name, value] for (name, (value, _)) in diagnostic_matrices_u]...)...,
        vcat([[name, value] for (name, (value, _)) in diagnostic_matrices_g]...)...,
    )
end

using Plots.Measures
gr(; size = (1.6 * 1200, 600), legend = true, bottom_margin = 10mm, left_margin = 10mm)
default(; titlefont = 20, legendfontsize = 12, guidefont = 14, tickfont = 14)

trunc = 15
trunc = min(trunc, input_dim, output_dim)
# color names in https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl

alg = LinearAlgebra.QRIteration()
plots = map([:in, :out]) do in_or_out
    diagnostics = in_or_out == :in ? all_diagnostic_matrices_u : all_diagnostic_matrices_g
    ref = in_or_out == :in ? "Hu_0.0_ekp_perfect" : "Hg_0.0_ekp_perfect"

    p = plot(; title = "Similarity of spectrum of $(in_or_out)put diagnostic", xlabel = "SV index")
    for (name, (mats, color)) in diagnostics
        svds = [svd(mat; alg) for mat in mats]
        sims = [cossim_cols(s.V, svd(ref_diag; alg).V) for (s, ref_diag) in zip(svds, diagnostics[ref][1])]
        name == ref ||
            plot!(p, mean(sims)[1:trunc]; ribbon = std(sims)[1:trunc], label = "sim ($ref vs. $name)", color)
        mean_S = mean([s.S[1:trunc] for s in svds])
        plot!(p, mean_S ./ mean_S[1]; label = "SVs ($name)", linestyle = :dash, linewidth = 3, color)
    end

    p
end
plot(plots...; layout = @layout([a b]))
savefig("figures/spectrum_comparison_$problem.png")
