module Truth

# Import Cloudy modules
using Cloudy.KernelTensors
using Cloudy.PDistributions
using Cloudy.Sources

# packages
using LinearAlgebra
using DifferentialEquations
using Sundials # CVODE_BDF() solver for ODE
using Random
using Distributions

export TruthObj
export run_cloudy_truth


#####
##### Structure definitions
#####

# Structure to organize the "truth"
struct TruthObj
    distr_init::PDistribution{Float64}
    solution::ODESolution # full solution of Cloudy run
    y_t::Array{Float64, 1} # true data (="observations")
    cov::Array{Float64, 2} # covariance of the obs noise
    data_names::Vector{String} 
end

# Constructors
function TruthObj(distr_init::PDistribution{Float64}, 
                  solution::ODESolution, cov::Array{Float64, 2},
                  data_names::Vector{String})

    y_t = vcat(solution.u'...)[end, :]

    TruthObj(distr_init, solution, y_t, cov, data_names)
end


#####
##### Function definitions
#####

"""
function run_cloudy_truth(kernel, dist, moments, tspan, data_names, obs_noise)

- `kernel` - Cloudy KernelTensor defining particle interactions
- `dist` - Cloudy.PDistribution defining the intial particle mass distirbution
- `moments` - Moments of dist to collect in Cloudy as they evolve over time,
              e.g. [0.0, 1.0, 2.0] for the 0th, 1st and 2nd moment
- `tspan` - Time period over which Cloudy is run, e.g. (0.0, 1.0)
- `data_names` - Names of the data/moments to collect, e.g. ["M0", "M1", "M2"]
- `obs_noise` - Covariance of the observational noise (assumed to be normally
                distributed with mean zero). Will be used as an attribute of 
                the returned TruthObj. If obs_noise is Nothing, it will be 
                defined within run_cloudy_truth based on draws from a normal
                distribution with mean 0 and standard deviation 2.

Returns a TruthObj.
"""
function run_cloudy_truth(kernel::KernelTensor{Float64}, 
                          dist::PDistribution{Float64}, 
                          moments::Array{Float64}, 
                          tspan::Tuple{Float64, Float64}, 
                          data_names::Vector{String},
                          obs_noise::Union{Array{Float64, 2}, Nothing};
                          rng_seed=1234)
    # Numerical parameters
    tol = 1e-7

    # moments_init is the initial condition for the ODE problem
    n_moments = length(moments)
    moments_init = fill(NaN, length(moments))
    for (i, mom) in enumerate(moments)
        moments_init[i] = PDistributions.moment(dist, convert(Float64, mom))
    end

    # Set up ODE problem: dM/dt = f(M,p,t)
    rhs(M, p, t) = get_int_coalescence(M, dist, kernel)
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
