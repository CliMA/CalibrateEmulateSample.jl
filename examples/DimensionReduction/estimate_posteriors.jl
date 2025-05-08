# Solve the problem with EKS


using Plots
using EnsembleKalmanProcesses
using Random
using JLD2

rng_seed = 41
rng = Random.MersenneTwister(rng_seed)

input_dim = 500
output_dim = 50

include("common_inverse_problem.jl")

n_trials = 10

r_in = 6
r_out = 1

in_diag = "Hu_ekp_prior"
out_diag = "Hg_ekp_prior"
Hu = diagnostic_mats[in_diag]
Hg = diagnostic_mats[out_diag]
@info "Diagnostic matrices = ($(in_diag), $(out_diag))"

if !isfile("ekp_1.jld2")
    include("generate_inverse_problem_data.jl") # will run n trials
end
if !isfile("diagnostic_matrices_1.jld2")
    include("build_an_compare_diagnostic_matrices.jl") # will run n trials
end

for trial = 1:n_trials

    # Load the EKP iterations
    loaded = load("ekp_$(trial).jld2")
    ekp = loaded["ekp"]
    prior = loaded["prior"]
    obs_noise_cov = loaded["obs_noise_cov"]
    y = loaded["y"]
    model = loaded["model"]
    input_dim = size(get_u(ekp,1),1)
    output_dim = size(get_g(ekp,1),1)

    prior_cov = cov(prior)
    prior_invrt = sqrt(inv(prior_cov))
    prior_rt = sqrt(prior_cov)
    obs_invrt = sqrt(inv(obs_noise_cov))
    obs_inv = inv(obs_noise_cov)
    
    # Load diagnostic container
    diagnostic_mats = load("diagnostic_matrices_$(trial).jld2")


    # [1] solve the problem with EKS - directly
    prior, y, obs_noise_cov, model, true_parameter = linear_exp_inverse_problem(input_dim, output_dim, rng)
    
    n_ensemble = 100
    n_iters_max = 50

    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    ekp = EnsembleKalmanProcess(initial_ensemble, y, obs_noise_cov, Sampler(prior); rng = rng)
    
    n_iters = [0]
    for i in 1:n_iters_max
        params_i = get_ϕ_final(prior, ekp)
        G_ens = hcat([forward_map(params_i[:, i], model) for i in 1:n_ensemble]...)
        terminate = update_ensemble!(ekp, G_ens)
        if !isnothing(terminate)
            n_iters[1] = i-1
            break
        end
    end

    ekp_u = get_u(ekp)
    ekp_g = get_g(ekp)
    
    # [2] Create emulator in truncated space, and run EKS on this
    min_iter = 
    max_iter = 8
    i_pairs = reduce(hcat, get_u(ekp)[min_iter:max_iter])
    o_pairs = reduce(hcat, get_g(ekp)[min_iter:max_iter])

    # Reduce space diagnostic matrix
    svdu = svd(Hu)

    #=
    # find by tolerance doesn't work well...
    tol = 0.999 # <1
    r_in_vec = accumulate(+,svdu.S) ./sum(svdu.S)
    r_in = sum(r_in_vec .< tol) + 1 # number evals needed for "tol" amount of information
    =#
    U_r = svdu.V[:,1:r_in]
    
    #=
    svdg = svd(Hg)
    r_out_vec = accumulate(+,svdg.S) ./sum(svdg.S)
    r_out = sum(r_out_vec .< tol) + 1 # number evals needed for "tol" amount of information
    =#
    V_r = svdg.V[:,1:r_out]
    
    X_r = U_r' * prior_invrt * i_pairs
    Y_r = V_r' * obs_invrt * o_pairs
    
    # true
    true_parameter = reshape(ones(input_dim),:,1)
    true_r = U_r' * prior_invrt * true_parameter
    y_r = V_r' * obs_invrt * y

    # [2a] exp-cubic model for regressor
    red_model_ids = ["expcubic1d","G"]
    red_model_id = red_model_ids[2]
    if red_model_id == "expcubic1d"
        Ylb = min(1, minimum(Y_r) - abs(mean(Y_r))) # some loose lower bound
        logY_r = log.(Y_r .- Ylb)
        β = logY_r / [ones(size(X_r)); X_r; X_r.^2; X_r.^3] # = ([1  X_r]' \ Y_r')'
        # invert relationship by
        # exp(β*X_r) + Ylb
        if r_in ==1 & r_out == 1  
            sc = scatter(X_r,logY_r)
            xmin = minimum(X_r)
            xmax = maximum(X_r)
            xrange = reshape(range(xmin,xmax,100), 1, :)
            expcubic = (β[4]*xrange.^3 + β[3]*xrange.^2 .+ β[2]*xrange .+ β[1])'
            plot!(sc, xrange, expcubic, legend=false)
            hline!(sc, [log.(y_r .- Ylb)])
            savefig(sc, "linreg_learn_scatter.png")
        end
    end       

    # now apply EKS to the new problem
    initial_ensemble = construct_initial_ensemble(rng, prior, n_ensemble)
    initial_r = U_r' * prior_invrt * initial_ensemble
    prior_r = ParameterDistribution(Samples(U_r' * prior_invrt * sample(rng,prior,1000)), repeat([no_constraint()], r_in), "prior_r")
    
    obs_noise_cov_r = V_r' * V_r # Vr' * invrt(noise) * noise * invrt(noise) * Vr
    ekp_r = EnsembleKalmanProcess(initial_r, y_r, obs_noise_cov_r, Sampler(mean(prior_r)[:], cov(prior_r)); rng = rng)
    
    n_iters = [0]
    for i in 1:n_iters_max
        params_i = get_ϕ_final(prior_r, ekp_r)
        if red_model_id == "expcubic1d"
            G_ens = exp.(β[4]*(params_i).^3 .+ β[3]*(params_i).^2 .+ β[2]*params_i .+ β[1]) .+ Ylb # use linear forward map in reduced space
        elseif red_model_id == "G"
            # uninformative eigenvectors
            sv_in = reduce(hcat, repeat([svdu.S], n_ensemble)) # repeat SVs, then replace first by params
            sv_in[1:size(params_i,1),:] = params_i
            # evaluate true G
            G_ens_full = reduce(hcat, [forward_map(prior_rt * svdu.V * sv, model) for sv in eachcol(sv_in)] )
            # project data back
            G_ens = V_r' * obs_invrt * G_ens_full
        end
           
        terminate = update_ensemble!(ekp_r, G_ens)
        if !isnothing(terminate)
            n_iters[1] = i-1
            break
        end
    end
    ekp_r_u = get_u(ekp_r)
    ekp_r_g = get_g(ekp_r)
    
    # map to same space: [here in reduced space first]
    spinup = 10*n_ensemble
    ekp_r_u = reduce(hcat,ekp_r_u)
    ekp_r_g = reduce(hcat,ekp_r_g)
    ekp_u = reduce(hcat, ekp_u)
    ekp_g = reduce(hcat, ekp_g)
    projected_ekp_u = U_r' * prior_invrt * ekp_u
    projected_ekp_g = V_r' * obs_invrt * ekp_g
    
    if r_in ==1 && r_out == 1
        pp1 = histogram(projected_ekp_u[:,spinup:end]', color= :gray, label="projected G", title="projected EKP samples (input)", legend=true)
        histogram!(pp1, ekp_r_u[:,spinup:end]', color = :blue, label="reduced")
        # pp1 = histogram(ekp_rlin_u[:,spinup:end]', color = :blue, label="reduced linear", yscale=:log10)

        pp2 = histogram(projected_ekp_g[:,spinup:end]', color= :gray, title ="projected EKP samples (output)")
        histogram!(pp2, ekp_r_g[:,spinup:end]', color = :blue, legend=false)
        #pp2 = histogram!(ekp_rlin_g[:,spinup:end]', color = :blue, legend=false, yscale=:log10)
        l = @layout [a b]
        pp = plot(pp1, pp2, layout=l)
        savefig(pp,"projected_histograms.png")
    end

    # compare in original space
    mean_final = get_u_mean_final(ekp)
    mean_final_in_red =  U_r' * prior_invrt * mean_final
    mean_red_final = get_u_mean_final(ekp_r)
    sv_in = svdu.S
    sv_in[1:r_in] = mean_red_final
    mean_red_final_full = prior_rt * svdu.V * sv_in
    
    @info """

    Reduced space dimension(input, output):  $((r_in, r_out))

    Norm of final-mean to true in reduced space:
    Using Full space optimization: $((1.0/r_in)*norm(mean_final_in_red - true_r)) 
    Using Red. space optimization: $((1.0/r_in)*norm(mean_red_final - true_r))

    Norm of final-mean to true in full space:
    Using Full space optimization: $((1.0/input_dim)*norm(mean_final - true_parameter))
    Using Red. space optimization: $((1.0/input_dim)*norm(mean_red_final_full - true_parameter))
    """
    #@info norm(cov(get_u_final(ekp),dims=2) - cov(get_u_final(ekp_r), dims=2))
    #=
    # 500 dim marginal plot..
    pp = plot(prior)

    spinup_iters = n_iters_max - 20
    posterior_samples = reduce(hcat,get_ϕ(prior,ekp)[spinup_iters:end]) # flatten over iterations (n_dim x n_particles)
    posterior_dist = ParameterDistribution(
        Dict(
            "distribution" => Samples(posterior_samples),
            "name" => "posterior samples",
            "constraint" => repeat([no_constraint()], input_dim),
        ),
    )
    plot!(pp, posterior_dist)
    =#
    
end
