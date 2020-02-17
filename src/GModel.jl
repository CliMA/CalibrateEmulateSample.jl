module GModel

using DocStringExtensions

using Random
using Sundials # CVODE_BDF() solver for ODE
using Distributions
using LinearAlgebra
using DifferentialEquations

export run_G
export run_G_ensemble

# TODO: It would be nice to have run_G_ensemble take a pointer to the
# function G (called run_G below), which maps the parameters u to G(u),
# as an input argument. In the example below, run_G runs Cloudy, but in
# general run_G will be supplied by the user and run the specific model
# the user wants to do UQ with.
# So, the interface would ideally be something like:
#      run_G_ensemble(params, G) where G is user-defined
"""
    GSettings{FT<:AbstractFloat, KT, D}

Structure to hold all information to run the forward model G (Cloudy)

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GSettings{FT<:AbstractFloat, KT, D}
    "a kernel tensor"
    kernel::KT
    "a model distribution"
    dist::D
    "the moments of dist that model should return"
    moments::Array{FT, 1}
    "time period over which to run the model, e.g., `(0, 1)`"
    tspan::Tuple{FT, FT}
end


"""
  run_G_ensemble(params::Array{FT, 2}, settings::GSettings;
                 rng_seed=42)

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
        # run Cloudy with the current parameters, i.e., map θ to G(θ)
        g_ens[i, :] = run_G(params[i, :], settings, update_params, moment, get_src)
    end

    return g_ens
end


"""
  run_G(u::Array{FT, 1}, settings::GSettings)

  Return g_u = G(u), a vector of length N_data (=N_moments in the case
  of Cloudy)

  - `u` - parameter vector of length N_parameters
  - `settings` - a GSettings struct

"""
function run_G(u::Array{FT, 1},
               settings::GSettings{FT},
               update_params,
               moment,
               get_src) where {FT<:AbstractFloat}

    # generate the initial distribution
    dist = update_params(settings.dist, u)

    # Numerical parameters
    tol = FT(1e-7)

    # Make sure moments are up to date. mom0 is the initial condition for the
    # ODE problem
    moments = settings.moments
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = moment(dist, convert(FT, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    local sol
    let kernel=settings.kernel,dist=dist,tspan=settings.tspan,moments_init=moments_init,tol=tol
      rhs(M, p, t) = get_src(M, dist, kernel)
      prob = ODEProblem(rhs, moments_init, tspan)
      # Solve the ODE
      sol = solve(prob, CVODE_BDF(), alg_hints=[:stiff], reltol=tol, abstol=tol)
    end
    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]

    return moments_final
end

end # module
