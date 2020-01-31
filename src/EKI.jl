"""
    EKI

To construct an object to perform Ensemble Kalman updates
It also measures errors to the truth to assess convergence
"""
module EKI

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
export run_model
export run_model_ensemble
export update_ensemble!


#####
#####  Structure definitions
#####

# structure to organize data
struct EKIObj{FT<:AbstractFloat,I<:Int}
     u::Vector{Array{FT, 2}}
     unames::Vector{String}
     g_t::Vector{FT}
     cov::Array{FT, 2}
     N_ens::I
     g::Vector{Array{FT, 2}}
     error::Vector{FT}
end


#####
##### Function definitions
#####

# outer constructors
function EKIObj(parameters::Array{FT, 2}, parameter_names::Vector{String},
                t_mean, t_cov::Array{FT, 2}) where {FT<:AbstractFloat}

    # ensemble size
    N_ens = size(parameters)[1]
    I = typeof(N_ens)
    # parameters
    u = Array{FT, 2}[] # array of Matrix{FT}'s
    push!(u, parameters) # insert parameters at end of array (in this case just 1st entry)
    # observations
    g = Vector{FT}[]
    # error store
    error = []

    EKIObj{FT,I}(u, parameter_names, t_mean, t_cov, N_ens, g, error)
end


"""
    construct_initial_ensemble(priors, N_ens)

Constructs the initial parameters, by sampling N_ens samples from specified
prior distributions.
"""
function construct_initial_ensemble(N_ens::I, priors; rng_seed=42) where {I<:Int}
    N_params = length(priors)
    params = zeros(N_ens, N_params)
    # Ensuring reproducibility of the sampled parameter values
    Random.seed!(rng_seed)
    for i in 1:N_params
        prior_i = priors[i]
        params[:, i] = rand(prior_i, N_ens)
    end

    return params
end

function compute_error(eki)
    meang = dropdims(mean(eki.g[end], dims=1), dims=1)
    diff = eki.g_t - meang
    X = eki.cov \ diff # diff: column vector
    newerr = dot(diff, X)
    push!(eki.error, newerr)
end


function run_model_ensemble(kernel::KT,
                            dist::D,
                            params::Array{FT, 2},
                            moments::Array{FT, 1},
                            tspan::Tuple{FT, FT},
                            moment,
                            update_params,
                            get_src;
                            rng_seed=42) where {D,KT, FT<:AbstractFloat}

    N_ens = size(params, 1) # params is N_ens x N_params
    n_moments = length(moments)
    g_ens = zeros(N_ens, n_moments)

    Random.seed!(rng_seed)
    for i in 1:N_ens
        # run model with the current parameters, i.e., map θ to G(θ)
        g_ens[i, :] = run_model(params[i, :], kernel, dist, moments, moment, update_params, get_src, tspan)
    end
    return g_ens
end


"""
run_model(kernel, dist, moments, tspan)

- `kernel` - is the collision-coalescence kernel that determines the evolution
           of the droplet size distribution
- `dist` - is a mass distribution function
- `moments` - is an array defining the moments of dist model will compute
              over time (e.g, [0.0, 1.0, 2.0])
- `tspan` - is a tuple definint the time interval over which the model is run
"""
function run_model(params::Array{FT, 1},
                    kernel::KT,
                    dist::D,
                    moments::Array{FT, 1},
                    moment,
                    update_params,
                    get_src,
                    tspan=Tuple{FT, FT}) where {D,KT,FT<:AbstractFloat}

    # generate the initial distribution
    dist = update_params(dist, params)

    # Numerical parameters
    tol = 1e-7

    # Make sure moments are up to date. mom0 is the initial condition for the
    # ODE problem
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = moment(dist, convert(FT, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)
    # Solve the ODE
    sol = solve(prob, CVODE_BDF(), alg_hints=[:stiff], reltol=tol, abstol=tol)
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]
    time = tspan[2]

    return moments_final
end


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

end

end # module EKI
