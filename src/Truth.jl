module Truth

# packages
using LinearAlgebra
using DifferentialEquations
using Sundials # CVODE_BDF() solver for ODE
using Random
using Distributions

export TruthObj
export run_model_truth


#####
##### Structure definitions
#####

# Structure to organize the "truth"
struct TruthObj{FT,D}
    distr_init::D
    solution::ODESolution # full solution of the model run
    y_t::Array{FT, 1} # true data (="observations")
    cov::Array{FT, 2} # covariance of the obs noise
    data_names::Vector{String}
end

# Constructors
function TruthObj(distr_init::D,
                  solution::ODESolution,
                  cov::Array{FT, 2},
                  data_names::Vector{String}) where {FT,D}

    y_t = vcat(solution.u'...)[end, :]

    TruthObj{FT,D}(distr_init, solution, y_t, cov, data_names)
end


#####
##### Function definitions
#####

"""
    run_model_truth(kernel::KT,
                     dist::D,
                     moments::Array{Float64},
                     tspan::Tuple{Float64, Float64},
                     data_names::Vector{String},
                     obs_noise::Union{Array{Float64, 2}, Nothing}
                     moment,
                     get_src;
                     rng_seed=1234) where {D,KT}

- `kernel` - Model KernelTensor defining particle interactions
- `dist` - Distribution defining the initial particle mass distribution
- `moments` - Moments of dist to collect in the model as they evolve over time,
              e.g. [0.0, 1.0, 2.0] for the 0th, 1st and 2nd moment
- `tspan` - Time period over which the model is run, e.g. (0.0, 1.0)
- `data_names` - Names of the data/moments to collect, e.g. ["M0", "M1", "M2"]
- `obs_noise` - Covariance of the observational noise (assumed to be normally
                distributed with mean zero). Will be used as an attribute of
                the returned TruthObj. If obs_noise is Nothing, it will be
                defined within run_model_truth based on draws from a normal
                distribution with mean 0 and standard deviation 2.

Returns a TruthObj.
"""
function run_model_truth(kernel::KT,
                          dist::D,
                          moments::Array{Float64},
                          tspan::Tuple{Float64, Float64},
                          data_names::Vector{String},
                          obs_noise::Union{Array{Float64, 2}, Nothing},
                          moment,
                          get_src;
                          rng_seed=1234) where {D,KT}
    # Numerical parameters
    tol = 1e-7

    # moments_init is the initial condition for the ODE problem
    n_moments = length(moments)
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = moment(dist, convert(Float64, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_src(M, dist, kernel)
    prob = ODEProblem(rhs, moments_init, tspan)

    # Solve the ODE
    sol = solve(prob, CVODE_BDF(), reltol=tol, abstol=tol, save_everystep=true)

    if obs_noise isa Nothing
        # Define observational noise
        # TODO: Write function to generate random SPD matrix
        Random.seed!(rng_seed)
        A = rand(Distributions.Normal(0, 2.0), n_moments, n_moments)
        obs_noise = A * A'
    end

    return TruthObj(dist, sol, obs_noise, data_names)
end

end # module Truth
