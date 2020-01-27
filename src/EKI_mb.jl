module EKI

"""
Module: EKI
-------------------------------------
Packages required: LinearAlgebra, 
                   Statistics,
                   Distributions
                   LinearAlgebra
                   DifferentialeEquations
                   Random
                   Sundials
-------------------------------------
Idea: To construct an object to perform Ensemble Kalman updates
      It also measures errors to the truth to assess convergence
-------------------------------------
Exports: EKIObject
         update_ensemble!
         compute_error
         construct_initial_ensemble
         run_cloudy
         run_cloudy_ensemble
-------------------------------------
"""

# Import Cloudy modules
using Cloudy.PDistributions
using Cloudy.Sources
using Cloudy.KernelTensors

# packages
using Random
using Statistics 
using Sundials # CVODE_BDF() solver for ODE
using Distributions
using LinearAlgebra
using DifferentialEquations

# exports
export EKIObj
export construct_initial_ensemble
export compute_error
export run_cloudy
export run_cloudy_ensemble
export update_ensemble!


#####
#####  Structure definitions
#####

# structure to organize data
struct EKIObj
     u::Vector{Array{Float64, 2}}
     unames::Vector{String}
     g_t::Vector{Float64}
     cov::Array{Float64, 2}
     N_ens::Int64
     g::Vector{Array{Float64, 2}}
     error::Vector{Float64}
end


#####
##### Function definitions
#####

# outer constructors
function EKIObj(parameters::Array{Float64, 2}, parameter_names::Vector{String},
                t_mean, t_cov::Array{Float64, 2})
    
    # ensemble size
    N_ens = size(parameters)[1]
    # parameters
    u = Array{Float64, 2}[] # array of Matrix{Float64}'s
    push!(u, parameters) # insert parameters at end of array (in this case just 1st entry)
    # observations
    g = Vector{Float64}[]
    # error store
    error = []
    
    EKIObj(u, parameter_names, t_mean, t_cov, N_ens, g, error)
end


"""
    construct_initial_ensemble(priors, N_ens)
  
Constructs the initial parameters, by sampling N_ens samples from specified 
prior distributions.
"""
function construct_initial_ensemble(N_ens::Int64, priors; rng_seed=42)
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i]
        params[:, i] = rand(prior_i, N_ens)
    end
    
    return params
end # function construct_initial ensemble

function compute_error(eki)
    meang = dropdims(mean(eki.g[end], dims=1), dims=1)
    diff = eki.g_t - meang 
    X = eki.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eki.error, newerr)
end # function compute_error


function run_cloudy_ensemble(kernel::KernelTensor{Float64}, 
                             dist::PDistribution{Float64},
                             params::Array{Float64, 2}, 
                             moments::Array{Float64, 1}, 
                             tspan::Tuple{Float64, Float64}; rng_seed=42) 

    N_ens = size(params, 1) # params is N_ens x N_params
    n_moments = length(moments)
    g_ens = zeros(N_ens, n_moments)

    Random.seed!(rng_seed)
    for i in 1:N_ens
        # run cloudy with the current parameters, i.e., map θ to G(θ)
        g_ens[i, :] = run_cloudy(params[i, :], kernel, dist, moments, tspan)
    end
    return g_ens
end # function run_cloudy_ensemble


"""
run_cloudy(kernel, dist, moments, tspan)

- `kernel` - is the collision-coalescence kernel that determines the evolution 
           of the droplet size distribution
- `dist` - is a mass distribution function
- `moments` - is an array defining the moments of dist Cloudy will compute
              over time (e.g, [0.0, 1.0, 2.0])
- `tspan` - is a tuple definint the time interval over which cloudy is run
"""
function run_cloudy(params::Array{Float64, 1}, kernel::KernelTensor{Float64}, 
                    dist::PDistributions.PDistribution{Float64}, 
                    moments::Array{Float64, 1}, 
                    tspan=Tuple{Float64, Float64})
  
    # generate the initial distribution
    dist = PDistributions.update_params(dist, params)

    # Numerical parameters
    tol = 1e-7

    # Make sure moments are up to date. mom0 is the initial condition for the 
    # ODE problem
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = PDistributions.moment(dist, convert(Float64, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src_coalescence(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)
    # Solve the ODE
    sol = solve(prob, CVODE_BDF(), alg_hints=[:stiff], reltol=tol, abstol=tol)
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]
    time = tspan[2]
        
    return moments_final
end # function run_cloudy


function update_ensemble!(eki, g)
    # u: N_ens x N_params
    u = eki.u[end]

    u_bar = fill(0.0, size(u)[2])
    # g: N_ens x N_data
    g_bar = fill(0.0, size(g)[2])
    
    cov_ug = fill(0.0, size(u)[2], size(g)[2])
    cov_gg = fill(0.0, size(g)[2], size(g)[2])

    # update means/covs with new param/observation pairs u, g
    for j = 1:eki.N_ens

        u_ens = u[j, :]
        g_ens = g[j, :]

        # add to mean
        u_bar += u_ens
        g_bar += g_ens
        
        #add to cov
        cov_ug += u_ens * g_ens' # cov_ug is N_params x N_data
        cov_gg += g_ens * g_ens'
    end
   
    u_bar = u_bar / eki.N_ens
    g_bar = g_bar / eki.N_ens
    cov_ug = cov_ug / eki.N_ens - u_bar * g_bar'
    cov_gg = cov_gg / eki.N_ens - g_bar * g_bar'

    # update the parameters (with additive noise too)
    noise = rand(MvNormal(zeros(size(g)[2]), eki.cov), eki.N_ens) # N_data * N_ens
    y = (eki.g_t .+ noise)' # add g_t (N_data) to each column of noise (N_data x N_ens), then transp. into N_ens x N_data
    tmp = (cov_gg + eki.cov) \ (y - g)' # N_data x N_data \ [N_ens x N_data - N_ens x N_data]' --> tmp is N_data x N_ens
    u += (cov_ug * tmp)' # N_ens x N_params
    
    # store new parameters (and observations)
    push!(eki.u, u) # N_ens x N_params
    push!(eki.g, g) # N_ens x N_data

    compute_error(eki)

end # function update_ensemble!

end # module EKI
