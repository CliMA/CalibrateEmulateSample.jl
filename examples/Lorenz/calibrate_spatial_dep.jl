# Import modules
using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Plots
using Random
using JLD2
using Statistics

# CES 
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.EnsembleKalmanProcesses.ParameterDistributions
using CalibrateEmulateSample.EnsembleKalmanProcesses.Localizers

const EKP = EnsembleKalmanProcesses

include("./lorenz_spatial_dep.jl")

########################################################################
############################ Problem setup #############################
########################################################################
rng_seed_init = 11
rng_i = MersenneTwister(rng_seed_init)

#Creating my sythetic data
#initalize model variables
nx = 40  #dimensions of parameter vector
gamma = 8 .+ 6 * sin.((4 * pi * range(0, stop = nx - 1, step = 1)) / nx)  #forcing (Needs to be of type EnsembleMemberConfig)
true_parameters = EnsembleMemberConfig(gamma)

dt = 0.01  #time step


# Spin up over T_long for an initial condition
T_long = 1000.0  #total time 
spinup_config = LorenzConfig(dt, T_long)
x_initial = randn(rng_i, nx)
x_spun_up = lorenz_solve(true_parameters, x_initial, spinup_config)
x0 = x_spun_up[:, end]  #last element of the run is the initial condition for creating the data


#Creating sythetic data
T = 24.0
ny = nx * 2   #number of data points
lorenz_config_settings = LorenzConfig(dt, T)

# construct how we compute Observations
T_start = T - 20.0
T_end = T
observation_config = ObservationConfig(T_start, T_end)

model_out_y = lorenz_forward(true_parameters, x0, lorenz_config_settings, observation_config)

#Observation covariance
n_samples = 200
shuffled_ids = shuffle(Int(floor(size(x_spun_up, 2) / 2)):size(x_spun_up, 2))
x_on_attractor = x_spun_up[:, shuffled_ids[1:n_samples]] # randomly select points from second half of spin up

y_ens = hcat(
    [
        lorenz_forward(true_parameters, x_on_attractor[:, j], lorenz_config_settings, observation_config) for
        j in 1:n_samples
    ]...,
)

# estimate noise as the variability of these pushed-forward samples on the attractor
obs_noise_cov = cov(y_ens, dims = 2) + 1e-2 * I
y = y_ens[:, 1]

#Prior covariance

pl = 4.0
psig = 5.0
B = zeros(nx, nx)
for ii in 1:nx
    for jj in 1:nx
        B[ii, jj] = psig^2 * exp(-abs(ii - jj) / pl)
    end
end
B_sqrt = sqrt(B)

#=
psig = 5.0
B = psig^2*I
B_sqrt = sqrt(B)
=#

#Prior mean
mu = 4.0 * ones(nx)

#Creating prior distribution
distribution = Parameterized(MvNormal(mu, B))
constraint = repeat([no_constraint()], nx)
name = "ml96_prior"

prior = ParameterDistribution(distribution, constraint, name)

########################################################################
############################# Running GNKI #############################
########################################################################

# EKP parameters
N_ens = 50
N_iter = 20

rng_seed = 2498

rng = MersenneTwister(rng_seed)

# initial parameters: N_params x N_ens
initial_params = construct_initial_ensemble(rng, prior, N_ens)

method = Inversion(prior)

@info "Ensemble size: $(N_ens)"
ekpobj = EKP.EnsembleKalmanProcess(initial_params, y, obs_noise_cov, method; rng = copy(rng), verbose = true)

count = 0
n_iter = N_iter
n_samples_exp = N_iter * N_ens
x_on_attractor = x_spun_up[:, shuffled_ids[1:n_samples_exp]] # randomly select points from second half of spin up
for i in 1:N_iter
    params_i = get_ϕ_final(prior, ekpobj)

    # If RMSE convergence criteria is not satisfied 
    G_ens = hcat(
        [
            lorenz_forward(
                EnsembleMemberConfig(params_i[:, j]),
                x_on_attractor[:, (i - 1) * N_ens + j],
                lorenz_config_settings,
                observation_config,
            ) for j in 1:N_ens
        ]...,
    )
    # Update 
    terminate = EKP.update_ensemble!(ekpobj, G_ens)
    if !isnothing(terminate)
        n_iter = i - 1
        break
    end
end
final_ensemble = get_ϕ_final(prior, ekpobj)

# Output figure save directory
homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

# Create plots

gr(size = (1.6 * 400, 400))
hm = heatmap(x_spun_up[:, (end - 10000):end], c = :viridis)
savefig(hm, joinpath(figure_save_directory, "spun_up_heatmap.png"))
savefig(hm, joinpath(figure_save_directory, "spun_up_heatmap.pdf"))

using Plots.Measures
gr(size = (2 * 1.6 * 600, 600), guidefontsize = 18, tickfontsize = 16, legendfontsize = 16)
p1 = plot(
    range(0, nx - 1, step = 1),
    [gamma mean(final_ensemble, dims = 2)],
    label = ["solution" "EKI"],
    color = [:black :lightgreen],
    linewidth = 4,
    xlabel = "Spatial index",
    ylabel = "Forcing (input)",
    left_margin = 15mm,
    bottom_margin = 15mm,
)

p2 = plot(
    1:length(y),
    y,
    ribbon = sqrt.(diag(get_obs_noise_cov(ekpobj))),
    label = "data",
    color = :black,
    linewidth = 4,
    xlabel = "Spatial index",
    ylabel = "State mean/std output)",
    left_margin = 15mm,
    bottom_margin = 15mm,
    xticks = (Int.(0:10:ny), [0, 10, 20, 30, (40, 0), 10, 20, 30, 40]),
)
plot!(p2, 1:length(y), get_g_mean_final(ekpobj), label = "mean-final-output", color = :lightgreen, linewidth = 4)

l = @layout [a b]
plt = plot(p1, p2, layout = l)

savefig(plt, figure_save_directory * "solution_spatial_dep_ens$(N_ens).png")

# save objects
save(joinpath(data_save_directory, "ekp_spatial_dep.jld2"), "ekpobj", ekpobj)
save(joinpath(data_save_directory, "priors_spatial_dep.jld2"), "prior", prior)

u_stored = EKP.get_u(ekpobj, return_array = false)
g_stored = EKP.get_g(ekpobj, return_array = false)


save(
    joinpath(data_save_directory, "calibrate_results_spatial_dep.jld2"),
    "inputs",
    u_stored,
    "outputs",
    g_stored,
    "truth_sample",
    y,
    "truth_sample_mean",
    mean(y_ens, dims = 2),
    "truth_input_constrained",
    gamma,
)
