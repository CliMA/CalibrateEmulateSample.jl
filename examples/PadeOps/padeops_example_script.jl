# Import modules
include(joinpath(@__DIR__, "..", "ci", "linkfig.jl"))
include(joinpath(@__DIR__, "padeops_gmodel.jl")) # Contains source code for PadeOps run

# Import modules
using Distributions  
using LinearAlgebra
using StatsPlots
using GaussianProcesses
using Plots
using Random
using JLD2
using Statistics
using Printf

# CES
using CalibrateEmulateSample.Emulators
using CalibrateEmulateSample.MarkovChainMonteCarlo
using CalibrateEmulateSample.Utilities
using CalibrateEmulateSample.EnsembleKalmanProcesses
using CalibrateEmulateSample.ParameterDistributions
using CalibrateEmulateSample.DataContainers
using CalibrateEmulateSample.Observations

rng_seed = 2413798
Random.seed!(rng_seed)






# Functions needed to standardize the output data to input into emulator
function get_standardizing_factors(data::Array{FT, 2}) where {FT}
    # Input: data size: N_data x N_ensembles
    # Ensemble median of the data
    norm_factor = median(data, dims = 2)
    return norm_factor
end

function get_standardizing_factors(data::Array{FT, 1}) where {FT}
    # Input: data size: N_data*N_ensembles (splatted)
    # Ensemble median of the data
    norm_factor = median(data)
    return norm_factor
end


# Output figure save directory
example_directory = @__DIR__
println(example_directory)
figure_save_directory = joinpath(example_directory, "output")
data_save_directory = joinpath(example_directory, "output")
if !isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if !isdir(data_save_directory)
    mkdir(data_save_directory)
end



###
### Constructing the truth data
###  Define the (true) parameters
###
# Define the parameters that we want to learn
z0_true = 2.5E-4     # nondimensional roughness length

params_true = [z0_true]                             # Vector{Float64}
param_names = ["z0"]                                #Vector{String}
n_param = length(param_names)
params_true = reshape(params_true, (n_param, 1))    # Matrix{Float64}

println(n_param)
println(params_true)


###
###  Define the parameter priors
###
# Lognormal prior or normal prior?
log_normal = false # THIS ISN't CURRENTLY IMPLEMENTED

function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2 / μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2 / μ^2)))
    return μ_log, σ_log
end

# Set prior distribution of parameter
prior_means = [z0_true]      # YIS: set prior means to the truth values? WHY?!
prior_stds = [2E-4]    # prior standard deviations
# YIS: creating dictionary variables for the two priors
prior_z0 = Dict(
    "distribution" => Parameterized(Normal(prior_means[1], prior_stds[1])),
    "constraint" => no_constraint(),
    "name" => param_names[1],
)
priors = ParameterDistribution([prior_z0])     # YIS: Note: ParameterDistribution takes in dictionaries as input

###
###  Define the data from which we want to learn the parameters
###
data_names = ["y0"]


### 
    
    
### Set up the forward map
### Settings to be inputted to PSettings
###
tstop = 100; # nondimensional time (I want to assign a time limit)
runid = 14;  # RunID for input file
tidx_start = 200;
delta_tidx = 200;
tidx_end = 800;
tidx_steps = Int((tidx_end - tidx_start + delta_tidx)/delta_tidx);
phi = 45; # latitude
omega = 7.29*10^(-5); # rad/s
L = 400;
G = 8;
Ro = G/(omega*L);
# Domain
Lx = 16; nx = 128;
Ly = 8; ny = 64;
Lz = 6; nz = 128;
stats_type = 5; # Stats type, which statistics to construct from the PadeOps data
# PadeOps input file to run and settings to apply to input file
inputfile_path = "/work2/09033/youngin/stampede2/PadeOps/problems/incompressible/neutral_pbl_files/"
inputfile_name = string("Run",@sprintf("%2.2i",runid),".dat");
inputfile = joinpath(inputfile_path, inputfile_name)
run(`cp $inputfile_path/Run00.dat $inputfile`);

# Constructs PSettings and PParams structures, see GModel.jl for the descriptions
padeops_settings = padeops_gmodel.PSettings(tstop, runid, tidx_start, delta_tidx, tidx_end, tidx_steps, phi, omega,
    L, G, Ro, Lx, Ly, Lz, nx, ny, nz, stats_type, inputfile);
padeops_params = padeops_gmodel.PParams(z0_true);

###
###  Now, to generate (artificial) truth samples
###  Note: The observables y are related to the parameters θ by: y = G(θ) + η
###
# PadeOps forward
# Input: params: [N_params, N_ens]
# Output: gt: [N_data, N_ens]
# For truth data, dropdims of the output since the forward model is only being run with N_ens=1
gt = dropdims(padeops_gmodel.run_G_ensemble(params_true, padeops_settings), dims = 2)



### To compute internal variability covariance from gt
### Then, add noise (covariance) intentionally to model predictions
### Prescribe variance or use a number of forward passes to define true interval variability
var_prescribe = true
n_samples = 10     # same as ensemble number
if var_prescribe == true     # right now, it is at false
    yt = zeros(length(gt), n_samples)
    noise_level = 0.05
    Γy = noise_level * convert(Array, Diagonal(gt))
    μ = zeros(length(gt))
    # Add Gaussian noise to model truth data to make a set of truth observations
    for i in 1:n_samples
        yt[:, i] = gt .+ rand(MvNormal(μ, Γy))     # YIS: this is noise added to truth run to get truth observations
    end
else
    println("Using truth values to compute covariance")
    yt = zeros(length(gt), n_samples)
    for i in 1:n_samples    # use a number of forward passes to define true interval covariance
        padeops_settings_local = padeops_gmodel.PSettings(tstop, runid, tidx_start, delta_tidx, tidx_end, tidx_steps, phi, omega,
    L, G, Ro, Lx, Ly, Lz, nx, ny, nz, stats_type, inputfile);
        yt[:, i] = padeops_gmodel.run_G_ensemble(params_true, padeops_settings_local)
    end
    # Covariance of truth data
    Γy = cov(yt, dims = 2)

    println(size(Γy), " ", rank(Γy))
end

# YIS: store data in Observations.Observation object
truth = Observations.Observation(yt, Γy, data_names)
# Truth sample for EKP
truth_sample = truth.mean



# CALIBRATE!!!
###
###  Calibrate: Ensemble Kalman Inversion
###

# settings for the forward model in the EKP
# Here, the forward model for the EKP settings can be set distinctly from the truth runs
padeops_settings_G = padeops_settings; # initialize to truth settings
padeops_settings_G.runid = padeops_settings_G.runid + 1;  # to differentiate from runid of truth run

# defining operations
log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

N_ens = 10 # number of ensemble members  
N_iter = 5 # number of EKI iterations  
# initial parameters: N_params x N_ens  (e.g., 1x10)
initial_params = construct_initial_ensemble(priors, N_ens; rng_seed = rng_seed)

# initialize EnsembleKalmanProcesses object      (e.g., 1x10)
ekiobj = EnsembleKalmanProcesses.EnsembleKalmanProcess(initial_params, truth_sample, truth.obs_noise_cov, Inversion())


# YIS: Solving the inverse problem in a loop
# EKI iterations
println("EKP inversion error:")
err = zeros(N_iter)
err_params = zeros(N_iter)
for i in 1:N_iter
    if log_normal == false
        params_i = get_u_final(ekiobj)
    else
        params_i = exp_transform(get_u_final(ekiobj))
    end
    # YIS: runs the GModels on the ensemble
    g_ens = padeops_gmodel.run_G_ensemble(params_i, padeops_settings_G)
    # YIS: updates the ensembles
    EnsembleKalmanProcesses.update_ensemble!(ekiobj, g_ens)
    err[i] = get_error(ekiobj)[end]
    err_params[i] = mean((params_true - mean(params_i, dims = 2)) .^ 2)
    println("Iteration: " * string(i) * ", Error (data): " * string(err[i]))
    println("Iteration: " * string(i) * ", Error (params): " * string(err_params[i]))
end

# EKI results: Has the ensemble collapsed toward the truth?
println("True parameters: ")
println(params_true)

println("\nEKI results:")
if log_normal == false
    println(mean(get_u_final(ekiobj), dims = 2))
else
    println(mean(exp_transform(get_u_final(ekiobj)), dims = 2))
end




