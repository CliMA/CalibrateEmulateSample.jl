module DynamicalModel

using DocStringExtensions

using Cloudy
using Cloudy.ParticleDistributions
using Cloudy.KernelTensors
using Cloudy.EquationTypes
using Cloudy.Coalescence

using DifferentialEquations

export ModelSettings
export run_dyn_model


"""
    ModelSettings{FT<:AbstractFloat, KT, D}

Structure to hold all information to run the dynamical model

# Fields
$(DocStringExtensions.FIELDS)
"""
struct ModelSettings{FT <: AbstractFloat, KT, D <: PrimitiveParticleDistribution{FT}}
    "a kernel tensor specifying the physics of collision-coalescence"
    kernel::KT
    "a cloud droplet mass distribution function"
    dist::D
    "the moments of `dist` that the model should return"
    moments::Array{FT, 1}
    "time period over which to run the model, e.g., `(0, 1)`"
    tspan::Tuple{FT, FT}
end


"""
    run_dyn_model(ϕ::Array{FT, 1}, settings::ModelSettings{FT}) where {FT<:AbstractFloat}

Run the dynamical model (Cloudy) for the given parameter vector ϕ=[N0,θ,k], which defines the initial distribution of droplet masses. This distribution 
is assumed to be a (scaled) gamma distribution with scaling factor N0 (defining the number of particles), scale parameter θ, and shape parameter k.
Return the model output, a vector of moments of the particle mass
distribution at the end time of the simulation.

 - `ϕ` - parameter vector 
 - `settings` - a ModelSettings struct

"""
function run_dyn_model(ϕ::Array{FT, 1}, settings::ModelSettings{FT}) where {FT <: AbstractFloat}

    # Generate the initial distribution
    dist_init = GammaPrimitiveParticleDistribution(ϕ...)
    moments_init = get_moments(dist_init)

    # Set up ODE problem: dM/dt = f(M, ϕ, t)
    tspan = (FT(0), FT(1))
    coalescence_coeff = 1 / 3.14 / 4 / 100
    kernel_func = x -> coalescence_coeff
    kernel = CoalescenceTensor(kernel_func, 0, FT(100))
    ODE_parameters = Dict(:dist => [dist_init], :kernel => settings.kernel, :dt => FT(1))

    rhs(m, par, t) = get_int_coalescence(OneModeCoalStyle(), m, par, par[:kernel])
    prob = ODEProblem(rhs, moments_init, settings.tspan, ODE_parameters)
    sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])

    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]

    return moments_final
end

end
