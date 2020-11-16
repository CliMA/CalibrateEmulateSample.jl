module GModel

using DocStringExtensions

# TODO: 
# 1) Define global variables or use the GSettings framework
# 2) Implement the forward pass in the run_G_ensemble framework
# 3) Finish validating Lorenz implementation
# 4) Test the Lorenz framework using CES
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
                        settings::GSettings{FT},
                        update_params,
                        moment,
                        get_src;
                        rng_seed=42) where {FT<:AbstractFloat}

    N_ens = size(params, 1) # params is N_ens x N_params
    n_moments = length(settings.moments)
    g_ens = zeros(N_ens, n_moments)

    Random.seed!(rng_seed)
    for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[i, :] = run_G(params[i, :], settings, update_params, moment, get_src)
    end

    return g_ens
end


"""
    run_G(u::Array{FT, 1},
          settings::GSettings{FT},
          update_params,
          moment,
          get_src) where {FT<:AbstractFloat}

Return g_u = G(u), a vector of length N_data.

 - `u` - parameter vector of length N_parameters
 - `settings` - a GSettings struct

"""
#function run_G(u::Array{FT, 1},
#               settings::GSettings{FT},
#               update_params,
#               moment,
#               get_src) where {FT<:AbstractFloat}
#
#    # generate the initial distribution
#    dist = update_params(settings.dist, u)
#
#    # Numerical parameters
#    tol = FT(1e-7)
#
#    # Make sure moments are up to date. mom0 is the initial condition for the
#    # ODE problem
#    moments = settings.moments
#    moments_init = fill(NaN, length(moments))
#    for (i, mom) in enumerate(moments)
#        moments_init[i] = moment(dist, convert(FT, mom))
#    end
#
#    # Set up ODE problem: dM/dt = f(M,p,t)
#    rhs(M, p, t) = get_src(M, dist, settings.kernel)
#    prob = ODEProblem(rhs, moments_init, settings.tspan)
#    # Solve the ODE
#    sol = solve(prob, CVODE_BDF(), alg_hints=[:stiff], reltol=tol, abstol=tol)
#    # Return moments at last time step
#    moments_final = vcat(sol.u'...)[end, :]
#
#    return moments_final
#end

function gyre_forward(params, x, y, t, y_avg)
	# Analytical quadruple gyre
	# # Kristy & Dabiri (2019) JFM
	# # Requires parameters: alpha, eps, omega
	# # Initialize
	if y_avg==true
		yens = zeros(size(params,1), length(x)*length(y)*2)
	else
		yens = zeros(size(params,1), length(x)*length(y)*length(t)*2)
	end
	for k in 1:size(params,1)
	    u = zeros(length(x), length(y), length(t)); v = u;
	    dt = t[2] - t[1];
	    # Unpack parameters
	    alpha = params[k,1];
	    eps = params[k,2];
	    omega = params[k,3];
	    for tt in 1:length(t);
	        a = eps*sin(omega*t[tt]);
	        b = 1 - 2*eps*sin(omega*t[tt]);
	        for i in 1:length(x);
	    	    for j in 1:length(y);
	                # Parameters
	    	        f = a*x[i].^2 + b*x[i];
	    	        u[i,j,tt] = -pi * alpha * sin(pi*f) * cos(pi*y[j]);
	    	        v[i,j,tt] = pi * alpha * cos(pi*f) * sin(pi*y[j]) * (2*a*x[i]+b);
	            end
                end
            end
	    if y_avg==true
            	u = mean(u,dims=3)
            	v = mean(v,dims=3)
	    end
	    yout = vcat(vcat(u...), vcat(v...))
	    yens[k,:] = yout
	end
    return yens
end

function lorenz_forward(settings, F, Fp, N, dt, tend, nstep)
	# run the Lorenz simulation
	xn, t = lorenz_solve(F, Fp, N, dt, tend, nstep)
	# Define statistics of interest
	if settings.stats_type == 1
		# Define averaging indices
		indices = findall(x -> (x>settings.t_start) 
				  && (x<settings.t_start+settings.T), t)
		# Average
		gt = mean(xn[:,indices], dims=2)
	else 
		ArgumentError("Setting "*settings.stats_type*" not implemented.")	
	end
	return gt
end

function lorenz_solve(F, Fp, N, dt, tend, nstep)
	# Initialize
	xn = zeros(Float64, N, nstep)
	t = zeros(Float64, nstep)

	# Initial perturbation
	X = fill(Float64(0.), N)
	X = X + Fp
	for j in 1:nstep
		X = RK4(X,dt,N,F)
		xn[:,j] = X
		t[j] = dt*j
	end
	# Output
	return xn, t

end 

# Lorenz 96 system
# f = dx/dt
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

	return f
end

function RK4(xold, dt, N, F)
	# Predictor steps
	k1 = f(xold, N, F)
	k2 = f(xold + k1*dt/2., N, F)
	k3 = f(xold + k2*dt/2., N, F)
	k4 = f(xold + k3*dt, N, F)
	# Step
	xnew = xold + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
	return xnew
end

end # module
