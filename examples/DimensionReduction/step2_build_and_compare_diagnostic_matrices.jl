using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Statistics
using Distributions
using Plots
using JLD2
using Manopt, Manifolds

include("./settings.jl")

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

n_samples = step2_num_prior_samples

Hu_evals = []
Hg_evals = []
Hu_mean_evals = []
Hg_mean_evals = []
Hu_ekp_prior_evals = []
Hg_ekp_prior_evals = []
Hu_ekp_final_evals = []
Hg_ekp_final_evals = []

sim_Hu_means = []
sim_Hg_means = []
sim_G_samples = []
sim_U_samples = []
sim_Hu_ekp_prior = []
sim_Hg_ekp_prior = []
sim_Hu_ekp_final = []
sim_Hg_ekp_final = []
sim_Huy_ekp_final = []
sim_Hgy_ekp_final = []

for trial in 1:num_trials
    # Load the EKP iterations
    loaded = load("datafiles/ekp_$(problem)_$(trial).jld2")
    ekp = loaded["ekp"]
    prior = loaded["prior"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]
    input_dim = size(get_u(ekp, 1), 1)
    output_dim = size(get_g(ekp, 1), 1)

    prior_cov = cov(prior)
    prior_invrt = sqrt(inv(prior_cov))
    prior_rt = sqrt(prior_cov)
    obs_invrt = sqrt(inv(obs_noise_cov))
    obs_inv = inv(obs_noise_cov)

    # random samples
    prior_samples = sample(prior, n_samples)

    if has_jac(model)
        # [1a] Large-sample diagnostic matrices with perfect grad(Baptista et al 2022)
        @info "Construct good matrix ($(n_samples) samples of prior, perfect grad)"
        gradG_samples = jac_forward_map(prior_samples, model)
        Hu = zeros(input_dim, input_dim)
        Hg = zeros(output_dim, output_dim)

        for j in 1:n_samples
            Hu .+= 1 / n_samples * prior_rt * gradG_samples[j]' * obs_inv * gradG_samples[j] * prior_rt
            Hg .+= 1 / n_samples * obs_invrt * gradG_samples[j] * prior_cov * gradG_samples[j]' * obs_invrt
        end

        # [1b] One-point approximation at mean value, with perfect grad
        @info "Construct with mean value (1 sample), perfect grad"
        prior_mean_appr = mean(prior) # approximate mean
        gradG_at_mean = jac_forward_map(prior_mean_appr, model)[1]
        # NB the logpdf of the prior at the ~mean is 1805 so pdf here is ~Inf
        Hu_mean = prior_rt * gradG_at_mean' * obs_inv * gradG_at_mean * prior_rt
        Hg_mean = obs_invrt * gradG_at_mean * prior_cov * gradG_at_mean' * obs_invrt
    else
        Hu = Hu_mean = NaN * zeros(input_dim)
        Hg = Hg_mean = NaN * zeros(output_dim)
    end

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
    Cug = C_at_prior[(input_dim + 1):end, 1:input_dim]
    #    SL_gradG = (pinvCuu * Cug')' # approximates ∇G with ensemble.
    #    Hu_ekp_prior = prior_rt * SL_gradG' * obs_inv * SL_gradG * prior_rt 
    #    Hg_ekp_prior = obs_invrt * SL_gradG * prior_cov * SL_gradG' * obs_invrt 
    Hu_ekp_prior = Cuu_invrt * Cug' * obs_inv * Cug * Cuu_invrt
    Hg_ekp_prior = obs_invrt * Cug * pinvCuu * Cug' * obs_invrt

    # [2b] One-point approximation at mean value with SL grad
    @info "Construct with mean value final (1 sample), SL grad"
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

    myCug = Cug'
    Huy_ekp_final = N_ens \ Cuu_invrt * myCug*obs_inv'*sum( # TODO: Check if whitening is correct
       (y - gg) * (y - gg)' for gg in eachcol(g)
    )*obs_inv*myCug' * Cuu_invrt

    dim_g = size(g, 1)
    Vgy_ekp_final = zeros(dim_g, 0)
    num_vecs = 10
    @assert num_vecs ≤ dim_g
    for k in 1:num_vecs
        println("vector $k")
        counter = 0
        M = Grassmann(dim_g, 1)
        f = (_, v) -> begin
            counter += 1
            Vs = hcat(Vgy_ekp_final, vec(v))
            Γtildeinv = obs_inv - Vs*inv(Vs'*obs_noise_cov*Vs)*Vs'
            res = sum( # TODO: Check if whitening is correct
                norm((y-gg)' * obs_invrt * (I - Vs*Vs') * myCug' * Cuu_invrt)^2# * det(Vs'*obs_noise_cov*Vs)^(-1/2) * exp(0.5(y-gg)'*Γtildeinv*(y-gg))
                for gg in eachcol(g)
            )
            mod(counter, 100) == 1 && println("   iter $counter: $res")
            res
        end
        v00 = eigvecs(Hg_ekp_final)[:,k:k]
        v0 = [v00 + randn(dim_g, 1) / 10 for _ in 1:dim_g]
        v0 = [v0i / norm(v0i) for v0i in v0]
        bestvec = NelderMead(M, f, NelderMeadSimplex(v0); stopping_criterion=StopWhenPopulationConcentrated(5.0, 5.0))
        # Orthogonalize
        proj = bestvec - Vgy_ekp_final * (Vgy_ekp_final' * bestvec)
        bestvec = proj / norm(proj)

        Vgy_ekp_final = hcat(Vgy_ekp_final, bestvec)
    end
    Vgy_ekp_final = hcat(Vgy_ekp_final, randn(dim_g, dim_g - num_vecs))
    Hgy_ekp_final = Vgy_ekp_final * diagm(vcat(num_vecs:-1:1, zeros(dim_g - num_vecs))) * Vgy_ekp_final'


    # cosine similarity of evector directions
    svdHu = svd(Hu)
    svdHg = svd(Hg)
    svdHu_mean = svd(Hu_mean)
    svdHg_mean = svd(Hg_mean)
    svdHu_ekp_prior = svd(Hu_ekp_prior)
    svdHg_ekp_prior = svd(Hg_ekp_prior)
    svdHu_ekp_final = svd(Hu_ekp_final)
    svdHg_ekp_final = svd(Hg_ekp_final)
    svdHuy_ekp_final = svd(Huy_ekp_final)
    svdHgy_ekp_final = svd(Hgy_ekp_final)
    if has_jac(model)
        @info """

        samples -> mean 
        $(cossim_cols(svdHu.V, svdHu_mean.V)[1:3])
        $(cossim_cols(svdHg.V, svdHg_mean.V)[1:3])

        samples + deriv -> mean + (no deriv) prior
        $(cossim_cols(svdHu.V, svdHu_ekp_prior.V)[1:3])
        $(cossim_cols(svdHg.V, svdHg_ekp_prior.V)[1:3])

        samples + deriv -> mean + (no deriv) final
        $(cossim_cols(svdHu.V, svdHu_ekp_final.V)[1:3])
        $(cossim_cols(svdHg.V, svdHg_ekp_final.V)[1:3])

        mean+(no deriv): prior -> final
        $(cossim_cols(svdHu_ekp_prior.V, svdHu_ekp_final.V)[1:3])
        $(cossim_cols(svdHg_ekp_prior.V, svdHg_ekp_final.V)[1:3])

        y-aware -> samples
        $(cossim_cols(svdHu.V, svdHuy_ekp_final.V)[1:3])
        $(cossim_cols(svdHg.V, svdHgy_ekp_final.V)[1:3])
        """
    end
    push!(Hu_evals, svdHu.S)
    push!(Hg_evals, svdHg.S)
    push!(Hu_mean_evals, svdHu_mean.S)
    push!(Hg_mean_evals, svdHg_mean.S)
    push!(Hu_ekp_prior_evals, svdHu_ekp_prior.S)
    push!(Hg_ekp_prior_evals, svdHg_ekp_prior.S)
    push!(Hu_ekp_final_evals, svdHu_ekp_final.S)
    push!(Hg_ekp_final_evals, svdHg_ekp_final.S)
    if has_jac(model)
        push!(sim_Hu_means, cossim_cols(svdHu.V, svdHu_mean.V))
        push!(sim_Hg_means, cossim_cols(svdHg.V, svdHg_mean.V))
        push!(sim_Hu_ekp_prior, cossim_cols(svdHu.V, svdHu_ekp_prior.V))
        push!(sim_Hg_ekp_prior, cossim_cols(svdHg.V, svdHg_ekp_prior.V))
        push!(sim_Hu_ekp_final, cossim_cols(svdHu.V, svdHu_ekp_final.V))
        push!(sim_Hg_ekp_final, cossim_cols(svdHg.V, svdHg_ekp_final.V))
        push!(sim_Huy_ekp_final, cossim_cols(svdHu.V, svdHuy_ekp_final.V))
        push!(sim_Hgy_ekp_final, cossim_cols(svdHg.V, svdHgy_ekp_final.V))
    end

    # cosine similarity to output svd from samples
    G_samples = forward_map(prior_samples, model)'
    svdG = svd(G_samples) # nonsquare, so permuted so evectors are V
    svdU = svd(prior_samples')

    if has_jac(model)
        push!(sim_G_samples, cossim_cols(svdHg.V, svdG.V))
        push!(sim_U_samples, cossim_cols(svdHu.V, svdU.V))
    end

    save(
        "datafiles/diagnostic_matrices_$(problem)_$(trial).jld2",
        "Hu", Hu,
        "Hg", Hg,
        "Hu_mean", Hu_mean,
        "Hg_mean", Hg_mean,
        "Hu_ekp_prior", Hu_ekp_prior,
        "Hg_ekp_prior", Hg_ekp_prior,
        "Hu_ekp_final", Hu_ekp_final,
        "Hg_ekp_final", Hg_ekp_final,
        "Huy_ekp_final", Huy_ekp_final,
        "Hgy_ekp_final", Hgy_ekp_final,
        "svdU", svdU,
        "svdG", svdG,
    )
end

using Plots.Measures
gr(size = (1.6 * 1200, 600), legend = true, bottom_margin = 10mm, left_margin = 10mm)
default(titlefont = 20, legendfontsize = 12, guidefont = 14, tickfont = 14)

normal_Hg_evals = [ev ./ ev[1] for ev in Hg_evals]
normal_Hg_mean_evals = [ev ./ ev[1] for ev in Hg_mean_evals]
normal_Hg_ekp_prior_evals = [ev ./ ev[1] for ev in Hg_ekp_prior_evals]
normal_Hg_ekp_final_evals = [ev ./ ev[1] for ev in Hg_ekp_final_evals]

loaded1 = load("datafiles/ekp_$(problem)_1.jld2")
ekp_tmp = loaded1["ekp"]
input_dim = size(get_u(ekp_tmp, 1), 1)
output_dim = size(get_g(ekp_tmp, 1), 1)

truncation = 15
truncation = Int(minimum([truncation, input_dim, output_dim]))
# color names in https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl

pg = plot(
    1:truncation,
    mean(sim_Hg_means)[1:truncation],
    ribbon = (std(sim_Hg_means) / sqrt(num_trials))[1:truncation],
    color = :blue,
    label = "sim (samples v mean)",
    legend = false,
)

plot!(
    pg,
    1:truncation,
    mean(sim_Hg_ekp_prior)[1:truncation],
    ribbon = (std(sim_Hg_ekp_prior) / sqrt(num_trials))[1:truncation],
    color = :red,
    alpha = 0.3,
    label = "sim (samples v mean-no-der) prior",
)
plot!(
    pg,
    1:truncation,
    mean(sim_Hg_ekp_final)[1:truncation],
    ribbon = (std(sim_Hg_ekp_final) / sqrt(num_trials))[1:truncation],
    color = :gold,
    label = "sim (samples v mean-no-der) final",
)
plot!(
    pg,
    1:truncation,
    mean(sim_Hgy_ekp_final)[1:truncation],
    ribbon = (std(sim_Hgy_ekp_final) / sqrt(num_trials))[1:truncation],
    color = :purple,
    label = "sim (samples v y-aware) final",
)

plot!(pg, 1:truncation, mean(normal_Hg_evals)[1:truncation], color = :black, label = "normalized eval (samples)")
plot!(
    pg,
    1:truncation,
    mean(normal_Hg_mean_evals)[1:truncation],
    color = :black,
    alpha = 0.7,
    label = "normalized eval (mean)",
)

plot!(
    pg,
    1:truncation,
    mean(normal_Hg_ekp_prior_evals)[1:truncation],
    color = :black,
    alpha = 0.3,
    label = "normalized eval (mean-no-der)",
)

plot!(pg, 1:truncation, mean(normal_Hg_ekp_final_evals)[1:truncation], color = :black, alpha = 0.3)


plot!(
    pg,
    1:truncation,
    mean(sim_G_samples)[1:truncation],
    ribbon = (std(sim_G_samples) / sqrt(num_trials))[1:truncation],
    color = :green,
    label = "similarity (PCA)",
)

title!(pg, "Similarity of spectrum of output diagnostic")


normal_Hu_evals = [ev ./ ev[1] for ev in Hu_evals]
normal_Hu_mean_evals = [ev ./ ev[1] for ev in Hu_mean_evals]
normal_Hu_ekp_prior_evals = [ev ./ ev[1] for ev in Hu_ekp_prior_evals]
normal_Hu_ekp_final_evals = [ev ./ ev[1] for ev in Hu_ekp_final_evals]


pu = plot(
    1:truncation,
    mean(sim_Hu_means)[1:truncation],
    ribbon = (std(sim_Hu_means) / sqrt(num_trials))[1:truncation],
    color = :blue,
    label = "sim (samples v mean)",
)

plot!(pu, 1:truncation, mean(normal_Hu_evals)[1:truncation], color = :black, label = "normalized eval (samples)")
plot!(
    pu,
    1:truncation,
    mean(normal_Hu_mean_evals)[1:truncation],
    color = :black,
    alpha = 0.7,
    label = "normalized eval (mean)",
)
plot!(
    pu,
    1:truncation,
    mean(normal_Hu_ekp_prior_evals)[1:truncation],
    color = :black,
    alpha = 0.3,
    label = "normalized eval (mean-no-der)",
)
plot!(pu, 1:truncation, mean(normal_Hu_ekp_final_evals)[1:truncation], color = :black, alpha = 0.3)

plot!(
    pu,
    1:truncation,
    mean(sim_U_samples)[1:truncation],
    ribbon = (std(sim_U_samples) / sqrt(num_trials))[1:truncation],
    color = :green,
    label = "similarity (PCA)",
)

plot!(
    pu,
    1:truncation,
    mean(sim_Hu_ekp_prior)[1:truncation],
    ribbon = (std(sim_Hu_ekp_prior) / sqrt(num_trials))[1:truncation],
    color = :red,
    alpha = 0.3,
    label = "sim (samples v mean-no-der) prior",
)
plot!(
    pu,
    1:truncation,
    mean(sim_Hu_ekp_final)[1:truncation],
    ribbon = (std(sim_Hu_ekp_final) / sqrt(num_trials))[1:truncation],
    color = :gold,
    label = "sim (samples v mean-no-der) final",
)
plot!(
    pu,
    1:truncation,
    mean(sim_Huy_ekp_final)[1:truncation],
    ribbon = (std(sim_Huy_ekp_final) / sqrt(num_trials))[1:truncation],
    color = :purple,
    label = "sim (samples v y-aware) final",
)

title!(pu, "Similarity of spectrum of input diagnostic")

layout = @layout [a b]
p = plot(pu, pg, layout = layout)

savefig(p, "figures/spectrum_comparison_$problem.png")
