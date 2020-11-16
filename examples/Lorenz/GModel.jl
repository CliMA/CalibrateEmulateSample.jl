module GModel

using DocStringExtensions

# TODO: 
# 3) Finish validating Lorenz implementation
# 4) Test the Lorenz framework using CES
# 	a) CES performance seems to be poor for simple QOIs and is sensitive to the CES framework selections
# 	b) Worse performance with increased inversion steps
# 5) Implement a periodic forcing and validate
# 6) Define statistics in periodic setting using filtering (optimal filtering?)
# 7) Test periodic Lorenz CES as a function of various statistics

using Random
#using Sundials # CVODE_BDF() solver for ODE
using Distributions
using LinearAlgebra
using DifferentialEquations

export run_G
export run_G_ensemble
export gyre_forward
export lorenz_forward

# TODO: It would be nice to have run_G_ensemble take a pointer to the
# function G (called run_G below), which maps the parameters u to G(u),
# as an input argument. In the example below, run_G runs the model, but
# in general run_G will be supplied by the user and run the specific
# model the user wants to do UQ with.
# So, the interface would ideally be something like:
#      run_G_ensemble(params, G) where G is user-defined
"""
    GSettings{FT<:AbstractFloat, KT, D}

Structure to hold all information to run the forward model G

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GSettings{FT<:AbstractFloat, KT, D}
    "model output"
    out::Array{FT, 1}
    "time period over which to run the model, e.g., `(0, 1)`"
    tspan::Tuple{FT, FT}
    "x domain"
    x::Array{FT, 1}
    "y domain"
    y::Array{FT, 1}
end

struct LSettings   
	# G model statistics type
	stats_type::Int32    
	# Integration time start
	t_start::Float64    
	# Integration length
	T::Float64
	# Initial perturbation
	Fp::Array{Float64}
	# Number of longitude steps
	N::Int32
	# Timestep
	dt::Float64
	# Simulation end time
	tend::Float64
end


"""
    run_G_ensemble(params::Array{FT, 2},
                   settings::GSettings{FT},
                   update_params,
                   moment,
                   get_src;
                   rng_seed=42) where {FT<:AbstractFloat}

Run the forward model G for an array of parameters by iteratively
calling run_G for each of the N_ensemble parameter values.
Return g_ens, an array of size N_ensemble x N_data, where
g_ens[j,:] = G(params[j,:])

 - `params` - array of size N_ensemble x N_parameters containing the
              parameters for which G will be run
 - `settings` - a GSetttings struct

"""
function run_G_ensemble(params::Array{FT, 2},
                        settings::LSettings,
                        rng_seed=42) where {FT<:AbstractFloat}
    # Initilize ensemble
    N_ens = size(params, 1) # params is N_ens x N_params
    if settings.stats_type == 1
	    nd = 2
	    #nd = settings.N # can average over N as well
    end
    g_ens = zeros(N_ens, nd)
    # Lorenz parameters
    #Fp = rand(Normal(0.0, 0.01), N); # Initial perturbation
    F = params[:, 1] # Forcing

    Random.seed!(rng_seed)
    for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
	g_ens[i, :] = lorenz_forward(settings, F[i]) 
    end

    return g_ens
end

# Forward pass of the Lorenz 96 model
# Inputs: settings: structure with stats_type, t_start, T
# F: scalar forcing Float64(1), Fp: initial perturbation Float64(N)
# N: number of longitude steps Int64(1), dt: time step Float64(1)
# tend: End of simulation Float64(1), nstep: 
function lorenz_forward(settings::LSettings, F) 
	# run the Lorenz simulation
	xn, t = lorenz_solve(settings, F)
	# Define statistics of interest
	if settings.stats_type == 1
		# Define averaging indices range
		indices = findall(x -> (x>settings.t_start) 
				  && (x<settings.t_start+settings.T), t)
		# Average in time and over longitude
		gtm = mean(mean(xn[:,indices], dims=2), dims=1)
		# Variance
		gtc = mean(mean((xn[:,indices].-gtm).^2, dims=2), dims=1)
		# Combine statistics
		gt = vcat(gtm, gtc)
	else 
		ArgumentError("Setting "*settings.stats_type*" not implemented.")	
	end
	return gt
end

# Solve the Lorenz 96 system 
# Inputs: F: scalar forcing Float64(1), Fp: initial perturbation Float64(N)
# N: number of longitude steps Int64(1), dt: time step Float64(1)
# tend: End of simulation Float64(1), nstep: 
function lorenz_solve(settings::LSettings, F)
	# Initialize
	nstep = Int32(settings.tend/settings.dt);
	xn = zeros(Float64, settings.N, nstep)
	t = zeros(Float64, nstep)
	# Initial perturbation
	X = fill(Float64(0.), settings.N)
	X = X + settings.Fp
	# March forward in time
	for j in 1:nstep
		X = RK4(X,settings.dt,settings.N,F)
		xn[:,j] = X
		t[j] = settings.dt*j
	end
	# Output
	return xn, t

end 

# Lorenz 96 system
# f = dx/dt
# Inputs: x: state, N: longitude steps, F: forcing
function f(x,N,F)
	f = zeros(Float64, N)
	# Loop over N positions
	for i in 3:N-1
		f[i] = -x[i-2]*x[i-1] + x[i-1]*x[i+1] - x[i] + F
	end
	# Periodic boundary conditions
	f[1] = -x[N-1]*x[N] + x[N]*x[2] - x[1] + F
	f[2] = -x[N]*x[1] + x[1]*x[3] - x[2] + F
	f[N] = -x[N-2]*x[N-1] + x[N-1]*x[1] - x[N] + F
	# Output
	return f
end

# RK4 solve
function RK4(xold, dt, N, F)
	# Predictor steps
	k1 = f(xold, N, F)
	k2 = f(xold + k1*dt/2., N, F)
	k3 = f(xold + k2*dt/2., N, F)
	k4 = f(xold + k3*dt, N, F)
	# Step
	xnew = xold + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
	# Output
	return xnew
end

end # module
