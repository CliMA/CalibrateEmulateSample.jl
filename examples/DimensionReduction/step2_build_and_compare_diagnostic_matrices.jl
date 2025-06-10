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
    mcmc_samples = loaded["mcmc_samples"]
    prior = loaded["prior"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]

    prior_cov = cov(prior)
    prior_invrt = sqrt(inv(prior_cov))
    prior_rt = sqrt(prior_cov)
    obs_invrt = sqrt(inv(obs_noise_cov))
    obs_inv = inv(obs_noise_cov)

    # random samples
    prior_samples = sample(prior, step2_num_prior_samples)

    # [1a] Large-sample diagnostic matrices with perfect grad(Baptista et al 2022)
    @info "Construct good matrix ($(step2_num_prior_samples) samples of prior, perfect grad)"
    gradG_samples = jac_forward_map(prior_samples, model)
    Hu = zeros(input_dim, input_dim)
    Hg = zeros(output_dim, output_dim)

    for j in 1:step2_num_prior_samples
        Hu .+= step2_num_prior_samples \ prior_rt * gradG_samples[j]' * obs_inv * gradG_samples[j] * prior_rt
        Hg .+= step2_num_prior_samples \ obs_invrt * gradG_samples[j] * prior_cov * gradG_samples[j]' * obs_invrt
    end

    diagnostic_matrices_u["Hu"] = Hu, :black
    diagnostic_matrices_g["Hg"] = Hg, :black

    # [1b] One-point approximation at mean value, with perfect grad
    @info "Construct with mean value (1 sample), perfect grad"
    prior_mean_appr = mean(prior) # approximate mean
    gradG_at_mean = jac_forward_map(reshape(prior_mean_appr, input_dim, 1), model)[1]
    # NB the logpdf of the prior at the ~mean is 1805 so pdf here is ~Inf
    Hu_mean = prior_rt * gradG_at_mean' * obs_inv * gradG_at_mean * prior_rt
    Hg_mean = obs_invrt * gradG_at_mean * prior_cov * gradG_at_mean' * obs_invrt

    diagnostic_matrices_u["Hu_mean"] = Hu_mean, :blue
    diagnostic_matrices_g["Hg_mean"] = Hg_mean, :blue

    # [2a] One-point approximation at mean value with SL grad
    @info "Construct with mean value prior (1 sample), SL grad"
    g = get_g(ekp, 1)
    u = get_u(ekp, 1)
    N_ens = get_N_ens(ekp)
    C_at_prior = cov([u; g], dims = 2) # basic cross-cov
    Cuu = C_at_prior[1:input_dim, 1:input_dim]
    svdCuu = svd(Cuu)
    nz = min(N_ens - 1, input_dim) # nonzero sv's
    pinvCuu = svdCuu.U[:, 1:nz] * Diagonal(1 ./ svdCuu.S[1:nz]) * svdCuu.Vt[1:nz, :] # can replace with localized covariance
    Cuu_invrt = svdCuu.U * Diagonal(1 ./ sqrt.(svdCuu.S)) * svdCuu.Vt
    Cug = C_at_prior[(input_dim + 1):end, 1:input_dim] # TODO: Isn't this Cgu?
    #    SL_gradG = (pinvCuu * Cug')' # approximates ∇G with ensemble.
    #    Hu_ekp_prior = prior_rt * SL_gradG' * obs_inv * SL_gradG * prior_rt 
    #    Hg_ekp_prior = obs_invrt * SL_gradG * prior_cov * SL_gradG' * obs_invrt 
    Hu_ekp_prior = Cuu_invrt * Cug' * obs_inv * Cug * Cuu_invrt
    Hg_ekp_prior = obs_invrt * Cug * pinvCuu * Cug' * obs_invrt

    diagnostic_matrices_u["Hu_ekp_prior"] = Hu_ekp_prior, :red
    diagnostic_matrices_g["Hg_ekp_prior"] = Hg_ekp_prior, :red

    println("Relative gradient error: ", norm(gradG_at_mean - (Cug * pinvCuu)) / norm(gradG_at_mean))

    # [2b] One-point approximation at mean value with SL grad
    @info "Construct with mean value EKP final (1 sample), SL grad"
    final_it = length(get_g(ekp))
    g = get_g(ekp, final_it)
    u = get_u(ekp, final_it)
    C_at_final = cov([u; g], dims = 2) # basic cross-cov
    Cuu = C_at_final[1:input_dim, 1:input_dim]
    svdCuu = svd(Cuu)
    nz = min(N_ens - 1, input_dim) # nonzero sv's
    pinvCuu = svdCuu.U[:, 1:nz] * Diagonal(1 ./ svdCuu.S[1:nz]) * svdCuu.Vt[1:nz, :] # can replace with localized covariance
    Cuu_invrt = svdCuu.U * Diagonal(1 ./ sqrt.(svdCuu.S)) * svdCuu.Vt
    Cug = C_at_final[(input_dim + 1):end, 1:input_dim] # TODO: Isn't this Cgu?
    #    SL_gradG = (pinvCuu * Cug')' # approximates ∇G with ensemble.
    #    Hu_ekp_final = prior_rt * SL_gradG' * obs_inv * SL_gradG * prior_rt # here still using prior roots not Cuu
    #    Hg_ekp_final = obs_invrt * SL_gradG * prior_cov * SL_gradG' * obs_invrt 
    Hu_ekp_final = Cuu_invrt * Cug' * obs_inv * Cug * Cuu_invrt
    Hg_ekp_final = obs_invrt * Cug * pinvCuu * Cug' * obs_invrt

    diagnostic_matrices_u["Hu_ekp_final"] = Hu_ekp_final, :gold
    diagnostic_matrices_g["Hg_ekp_final"] = Hg_ekp_final, :gold

    @info "Construct y-informed at EKP final (SL grad)"
    myCug = Cug'
    Huy_ekp_final = N_ens \ Cuu_invrt * myCug*obs_inv'*sum( # TODO: Check if whitening is correct
       (y - gg) * (y - gg)' for gg in eachcol(g)
    )*obs_inv*myCug' * Cuu_invrt

    dim_g = size(g, 1)
    Vgy_ekp_final = zeros(dim_g, 0)
    num_vecs = 1
    @assert num_vecs ≤ dim_g
    for k in 1:num_vecs
        println("vector $k")
        counter = 0
        M = Grassmann(dim_g, 1)
        f = (_, v) -> begin
            counter += 1
            Vs = hcat(Vgy_ekp_final, vec(v))
            Γtildeinv = obs_inv - Vs*inv(Vs'*obs_noise_cov*Vs)*Vs'
            res = N_ens \ sum( # TODO: Check if whitening is correct
                norm((y-gg)' * obs_invrt * (I - Vs*Vs') * myCug' * Cuu_invrt)^2# * det(Vs'*obs_noise_cov*Vs)^(-1/2) * exp(0.5(y-gg)'*Γtildeinv*(y-gg))
                for gg in eachcol(g)
            )
            mod(counter, 100) == 1 && println("   iter $counter: $res")
            res
        end
        v00 = eigvecs(Hg_ekp_final)[:,k:k]
        v0 = [v00 + randn(dim_g, 1) / 10 for _ in 1:dim_g]
        v0 = [v0i / norm(v0i) for v0i in v0]
        bestvec = NelderMead(M, f, NelderMeadSimplex(v0); stopping_criterion=StopWhenPopulationConcentrated(5000.0, 5000.0)) # TODO: Set very high to effectively turn off this diagnostic for speed
        # Orthogonalize
        proj = bestvec - Vgy_ekp_final * (Vgy_ekp_final' * bestvec)
        bestvec = proj / norm(proj)

        Vgy_ekp_final = hcat(Vgy_ekp_final, bestvec)
    end
    Vgy_ekp_final = hcat(Vgy_ekp_final, randn(dim_g, dim_g - num_vecs))
    Hgy_ekp_final = Vgy_ekp_final * diagm(vcat(num_vecs:-1:1, zeros(dim_g - num_vecs))) * Vgy_ekp_final'

    diagnostic_matrices_u["Huy_ekp_final"] = Huy_ekp_final, :purple
    diagnostic_matrices_g["Hgy_ekp_final"] = Hgy_ekp_final, :purple


    @info "Construct with mean value MCMC final (1 sample), SL grad"
    u = mcmc_samples
    g = hcat([forward_map(uu, model) for uu in eachcol(u)]...)
    N_ens = size(u, 2)
    C_at_final = cov([u; g], dims = 2) # basic cross-cov
    Cuu = C_at_final[1:input_dim, 1:input_dim]
    svdCuu = svd(Cuu)
    nz = min(N_ens - 1, input_dim) # nonzero sv's
    pinvCuu = svdCuu.U[:, 1:nz] * Diagonal(1 ./ svdCuu.S[1:nz]) * svdCuu.Vt[1:nz, :] # can replace with localized covariance
    Cuu_invrt = svdCuu.U * Diagonal(1 ./ sqrt.(svdCuu.S)) * svdCuu.Vt
    Cug = C_at_final[(input_dim + 1):end, 1:input_dim] # TODO: Isn't this Cgu?
    Hu_mcmc_final = Cuu_invrt * Cug' * obs_inv * Cug * Cuu_invrt
    Hg_mcmc_final = obs_invrt * Cug * pinvCuu * Cug' * obs_invrt

    diagnostic_matrices_u["Hu_mcmc_final"] = Hu_mcmc_final, :green
    diagnostic_matrices_g["Hg_mcmc_final"] = Hg_mcmc_final, :green

    @info "Construct y-informed at MCMC final (perfect grad)"
    gradG_samples = jac_forward_map(u, model)
    Huy = zeros(input_dim, input_dim)

    for j in 1:N_ens
        Huy .+= 1 / N_ens * prior_rt * gradG_samples[j]' * obs_inv^2 * (y - g[:, j]) * (y - g[:, j])' * obs_inv^2 * gradG_samples[j] * prior_rt # TODO: Is the obs_inv^2 correct?
    end

    @info "Construct y-informed at MCMC final (SL grad)"
    myCug = Cug'
    Huy_mcmc_final = N_ens \ Cuu_invrt * myCug*obs_inv'*sum( # TODO: Check if whitening is correct
       (y - gg) * (y - gg)' for gg in eachcol(g)
    )*obs_inv*myCug' * Cuu_invrt

    dim_g = size(g, 1)
    Vgy_mcmc_final = zeros(dim_g, 0)
    num_vecs = 1
    @assert num_vecs ≤ dim_g
    for k in 1:num_vecs
        println("vector $k")
        counter = 0
        M = Grassmann(dim_g, 1)
        f = (_, v) -> begin
            counter += 1
            Vs = hcat(Vgy_mcmc_final, vec(v))
            Γtildeinv = obs_inv - Vs*inv(Vs'*obs_noise_cov*Vs)*Vs'
            res = N_ens \ sum( # TODO: Check if whitening is correct
                norm((y-gg)' * obs_invrt * (I - Vs*Vs') * myCug' * Cuu_invrt)^2# * det(Vs'*obs_noise_cov*Vs)^(-1/2) * exp(0.5(y-gg)'*Γtildeinv*(y-gg))
                for gg in eachcol(g)
            )
            mod(counter, 100) == 1 && println("   iter $counter: $res")
            res
        end
        v00 = eigvecs(Hg_mcmc_final)[:,k:k]
        v0 = [v00 + randn(dim_g, 1) / 10 for _ in 1:dim_g]
        v0 = [v0i / norm(v0i) for v0i in v0]
        bestvec = NelderMead(M, f, NelderMeadSimplex(v0); stopping_criterion=StopWhenPopulationConcentrated(5000.0, 5000.0)) # TODO: Set very high to effectively turn off this diagnostic for speed
        # Orthogonalize
        proj = bestvec - Vgy_mcmc_final * (Vgy_mcmc_final' * bestvec)
        bestvec = proj / norm(proj)

        Vgy_mcmc_final = hcat(Vgy_mcmc_final, bestvec)
    end
    Vgy_mcmc_final = hcat(Vgy_mcmc_final, randn(dim_g, dim_g - num_vecs))
    Hgy_mcmc_final = Vgy_mcmc_final * diagm(vcat(num_vecs:-1:1, zeros(dim_g - num_vecs))) * Vgy_mcmc_final'

    diagnostic_matrices_u["Huy_mcmc_final"] = Huy_mcmc_final, :orange
    diagnostic_matrices_g["Hgy_mcmc_final"] = Hgy_mcmc_final, :orange

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
gr(; size=(1.6 * 1200, 600), legend=true, bottom_margin=10mm, left_margin=10mm)
default(; titlefont=20, legendfontsize=12, guidefont=14, tickfont=14)

trunc = 15
trunc = min(trunc, input_dim, output_dim)
# color names in https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl

alg = LinearAlgebra.QRIteration()
plots = map([:in, :out]) do in_or_out
    diagnostics = in_or_out == :in ? all_diagnostic_matrices_u : all_diagnostic_matrices_g
    ref = in_or_out == :in ? "Hu" : "Hg"

    p = plot(; title="Similarity of spectrum of $(in_or_out)put diagnostic", xlabel="SV index")
    for (name, (mats, color)) in diagnostics
        svds = [svd(mat; alg) for mat in mats]
        sims = [cossim_cols(s.V, svd(ref_diag; alg).V) for (s, ref_diag) in zip(svds, diagnostics[ref][1])]
        name == ref || plot!(p, mean(sims)[1:trunc]; ribbon=std(sims)[1:trunc], label="sim ($ref vs. $name)", color)
        mean_S = mean([s.S[1:trunc] for s in svds])
        plot!(p, mean_S ./ mean_S[1]; label="SVs ($name)", linestyle=:dash, linewidth=3, color)
    end

    p
end
plot(plots...; layout=@layout([a b]))
savefig("figures/spectrum_comparison_$problem.png")
