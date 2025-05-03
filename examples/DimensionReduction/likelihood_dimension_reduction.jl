using LinearAlgebra
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions
using Statistics
using Distributions
using Plots
using JLD2 
#Utilities
function cossim(x::VV1,y::VV2) where {VV1 <: AbstractVector, VV2 <: AbstractVector}
    return dot(x,y)/(norm(x)*norm(y))
end
function cossim_pos(x::VV1,y::VV2) where {VV1 <: AbstractVector, VV2 <: AbstractVector}
    return abs(cossim(x,y))
end
function cossim_cols(X::AM1, Y::AM2) where {AM1 <: AbstractMatrix, AM2 <: AbstractMatrix}
    return [cossim_pos(c1,c2) for (c1,c2) in zip(eachcol(X), eachcol(Y))]
end

n_samples = 2000 # paper uses 5e5
n_trials = 20 # get from generate_inverse_problem_data

if !isfile("ekp_1.jld2")
    include("generate_inverse_problem_data.jl") # will run n trials
else
    include("forward_maps.jl")
    loaded1 = load("ekp_1.jld2")
    ekp = loaded1["ekp"]
    prior = loaded1["prior"]
    obs_noise_cov = loaded1["obs_noise_cov"]
    input_dim = size(get_u(ekp,1),1)
    output_dim = size(get_g(ekp,1),1)
end

# new terminology
prior_cov = cov(prior)
prior_invrt = sqrt(inv(prior_cov))
prior_rt = sqrt(prior_cov)
obs_invrt = sqrt(inv(obs_noise_cov))
obs_inv = inv(obs_noise_cov)

Hu_evals = []
Hg_evals = []
Hu_mean_evals = []
Hg_mean_evals = []
Hu_ekp_prior_evals = []
Hg_ekp_prior_evals = []

sim_Hu_means = []
sim_Hg_means = []
sim_G_samples = []
sim_U_samples = []
sim_Hu_ekp_prior = []
sim_Hg_ekp_prior = []

for i = 1:n_trials

    # Load the EKP iterations
    loaded = load("ekp_$(i).jld2")
    ekp = loaded["ekp"]
    prior = loaded["prior"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]

    # random samples
    prior_samples = sample(prior, n_samples)
    
    # [1a] Large-sample diagnostic matrices with perfect grad(Baptista et al 2022)
    @info "Construct good matrix ($(n_samples) samples of prior, perfect grad)"
    gradG_samples = jac_forward_map(prior_samples, model)
    Hu = zeros(input_dim,input_dim)
    Hg = zeros(output_dim,output_dim)

    for j in 1:n_samples
        Hu .+=  1/n_samples * prior_rt * gradG_samples[j]' * obs_inv * gradG_samples[j] * prior_rt
        Hg .+=  1/n_samples * obs_invrt * gradG_samples[j] * prior_cov * gradG_samples[j]' * obs_invrt
    end
    
    # [1b] One-point approximation at mean value, with perfect grad
    @info "Construct with mean value (1 sample), perfect grad"
    prior_mean_appr = mean(prior) # approximate mean
    gradG_at_mean = jac_forward_map(prior_mean_appr, model)[1]
    # NB the logpdf of the prior at the ~mean is 1805 so pdf here is ~Inf
    Hu_mean = prior_rt * gradG_at_mean' * obs_inv * gradG_at_mean * prior_rt 
    Hg_mean = obs_invrt * gradG_at_mean * prior_cov * gradG_at_mean' * obs_invrt 

    # [2] One-point approximation at mean value with SL grad
    @info "Construct with mean value (1 sample), SL grad"
    g = get_g(ekp,1)
    u = get_u(ekp,1)
    N_ens = get_N_ens(ekp)
    C_at_prior = cov([u;g], dims=2) # basic cross-cov
    Cuu = C_at_prior[1:input_dim, 1:input_dim]
    svdCuu = svd(Cuu) 
    nz = min(N_ens-1, input_dim) # nonzero sv's
    pinvCuu = svdCuu.U[:,1:nz] * Diagonal(1 ./ svdCuu.S[1:nz]) * svdCuu.Vt[1:nz,:] # can replace with localized covariance
    Cug = C_at_prior[input_dim+1:end,1:input_dim]
    SL_gradG = (pinvCuu * Cug')' # approximates ∇G with ensemble.
    Hu_ekp_prior = prior_rt * SL_gradG' * obs_inv * SL_gradG * prior_rt 
    Hg_ekp_prior = obs_invrt * SL_gradG * prior_cov * SL_gradG' * obs_invrt 

    # cosine similarity of evector directions
    svdHu = svd(Hu)
    svdHg = svd(Hg)
    svdHu_mean = svd(Hu_mean)
    svdHg_mean = svd(Hg_mean)
    svdHu_ekp_prior = svd(Hu_ekp_prior)
    svdHg_ekp_prior = svd(Hg_ekp_prior)
    @info """

    samples -> mean value
    $(cossim_cols(svdHu.V, svdHu_mean.V)[1:3])
    $(cossim_cols(svdHg.V, svdHg_mean.V)[1:3])

    samples + deriv -> mean + (no deriv)
    $(cossim_cols(svdHu_mean.V, svdHu_ekp_prior.V)[1:3])
    $(cossim_cols(svdHg_mean.V, svdHg_ekp_prior.V)[1:3])

    """
    push!(sim_Hu_means, cossim_cols(svdHu.V, svdHu_mean.V))
    push!(sim_Hg_means, cossim_cols(svdHg.V, svdHg_mean.V))
    push!(Hu_evals, svdHu.S)
    push!(Hg_evals, svdHg.S)
    push!(Hu_mean_evals, svdHu_mean.S)
    push!(Hg_mean_evals, svdHg_mean.S)
    push!(Hu_ekp_prior_evals, svdHu_ekp_prior.S)
    push!(Hg_ekp_prior_evals, svdHg_ekp_prior.S)
    push!(sim_Hu_ekp_prior, cossim_cols(svdHu.V, svdHu_ekp_prior.V))
    push!(sim_Hg_ekp_prior, cossim_cols(svdHg.V, svdHg_ekp_prior.V))

    # cosine similarity to output svd from samples
    G_samples = forward_map(prior_samples, model)'
    svdG = svd(G_samples) # nonsquare, so permuted so evectors are V
    svdU = svd(prior_samples')
    
    push!(sim_G_samples, cossim_cols(svdHg.V, svdG.V))
    push!(sim_U_samples, cossim_cols(svdHu.V, svdU.V))
    
end

using Plots.Measures
gr(size=(1.6*1200,600), legend=true, bottom_margin = 10mm, left_margin = 10mm)
default(
    titlefont = 20,
    legendfontsize = 12,
    guidefont = 14,
    tickfont = 14,
)

normal_Hg_evals = [ev ./ ev[1] for ev in Hg_evals]
normal_Hg_mean_evals = [ev ./ ev[1] for ev in Hg_mean_evals]
normal_Hg_ekp_prior_evals = [ev ./ ev[1] for ev in Hg_ekp_prior_evals]

truncation = 15
truncation = Int(minimum([truncation,input_dim, output_dim]))

#= plot(
    1:truncation,
    (mean(sim_Hg_means) .* mean(normal_Hg_evals))[1:truncation],
    ribbon = (std(sim_Hg_means) .* mean(normal_Hg_evals)/sqrt(n_trials))[1:truncation],
    color = :red,
    label = "sim (samples v mean)",

)
=#
pg = plot(
    1:truncation,
    mean(sim_Hg_means)[1:truncation], 
    ribbon = (std(sim_Hg_means)/sqrt(n_trials))[1:truncation],
    color = :blue,
    label = "sim (samples v mean)",
    legend=false,
)

plot!(
    pg,
    1:truncation,
    mean(sim_Hg_ekp_prior)[1:truncation], 
    ribbon = (std(sim_Hg_ekp_prior)/sqrt(n_trials))[1:truncation],
    color = :red,
    label = "sim (samples v mean-no-der)",
)

plot!(
    pg,
    1:truncation,
    mean(normal_Hg_evals)[1:truncation],
    color = :black,
    label = "normalized eval (samples)",
)
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

plot!(
    pg,
    1:truncation,
    mean(sim_G_samples)[1:truncation], 
    ribbon = (std(sim_G_samples)/sqrt(n_trials))[1:truncation],
    color = :green,
    label = "similarity (PCA)",
)

title!(pg, "Similarity of spectrum of output diagnostic")


normal_Hu_evals = [ev ./ ev[1] for ev in Hu_evals]
normal_Hu_mean_evals = [ev ./ ev[1] for ev in Hu_mean_evals]
normal_Hu_ekp_prior_evals = [ev ./ ev[1] for ev in Hu_ekp_prior_evals]

#= plot(
    1:truncation,
    (mean(sim_Hu_means) .* mean(normal_Hu_evals))[1:truncation],
    ribbon = (std(sim_Hu_means) .* mean(normal_Hu_evals)/sqrt(n_trials))[1:truncation],
    color = :red,
    label = "similarity scaled by eval",
)=#

pu = plot(
    1:truncation,
    mean(sim_Hu_means)[1:truncation], 
    ribbon = (std(sim_Hu_means)/sqrt(n_trials))[1:truncation],
    color = :blue,
    label = "sim (samples v mean)",
)

plot!(
    pu,
    1:truncation,
    mean(normal_Hu_evals)[1:truncation],
    color = :black,
    label = "normalized eval (samples)",
)
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
plot!(
    pu,
    1:truncation,
    mean(sim_U_samples)[1:truncation], 
    ribbon = (std(sim_U_samples)/sqrt(n_trials))[1:truncation],
    color = :green,
    label = "similarity (PCA)",
)

plot!(
    pu,
    1:truncation,
    mean(sim_Hu_ekp_prior)[1:truncation], 
    ribbon = (std(sim_Hu_ekp_prior)/sqrt(n_trials))[1:truncation],
    color = :red,
    label = "sim (samples v mean-no-der)",
)

title!(pu, "Similarity of spectrum of input diagnostic")

layout = @layout [a b]
p = plot(pu, pg, layout = layout)

savefig(p, "spectrum_comparison.png")
     

