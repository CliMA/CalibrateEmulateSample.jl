module DynamicalModel

using DocStringExtensions

using Cloudy
using Cloudy.ParticleDistributions
using Cloudy.KernelTensors
using Cloudy.EquationTypes
using Cloudy.Coalescence

using Sundials # CVODE_BDF() solver for ODE


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

Run the dynamical model (Cloudy) for the given parameter vector ϕ. 
Return the model output, a vector of moments of the particle mass
distribution at the end time of the simulation.

 - `ϕ` - parameter vector of length N_parameters
 - `settings` - a ModelSettings struct

"""
function run_dyn_model(ϕ::Array{FT, 1}, settings::ModelSettings{FT}) where {FT <: AbstractFloat}

    # Generate the initial distribution
    # TODO: Obviously the GammaPrimitiveParticleDistribution should not be
    # hard-coded here. Since update_params has been removed from
    # ParticleDistributions.jl, I tried the following:
    # Determine the type of the distribution, then create a new instance of
    # that type using the desired parameters. This doesn't work; I get an
    # error saying LoadError: MethodError: no method matching GammaPrimitiveParticleDistribution{Float64}(::Float64, ::Float64, ::Float64)

    #dist_init_type = typeof(settings.dist)
    #dist_init = dist_init_type(ϕ...)
    dist_init = GammaPrimitiveParticleDistribution(ϕ...)
    moments_init = get_moments(dist_init)
    #dist_init = settings.dist(ϕ...)


#    ################ below is the "original" setup" 
#    # Numerical parameters
     tol = FT(1e-7)
#
#    # Set up ODE problem: dM/dt = f(M, ϕ, t)
#    rhs(M, par, t) = get_int_coalescence(M, par, settings.kernel)
#    ODE_par = Dict(:dist => dist_init)  # ODE parameters
#    prob = ODEProblem(rhs, moments_init, settings.tspan, ODE_par)
#    # Solve the ODE
#    sol = solve(prob, CVODE_BDF(), alg_hints = [:stiff], reltol = tol, abstol = tol)
#    #
#    ################ above is the "original" setup


    ################ below is the new setup (with unreleased Cloudy code)
    # Set up ODE problem: dM/dt = f(M, ϕ, t)
    tspan = (FT(0), FT(1))
    #kernel_func = x -> 5e-3 * (x[1] + x[2])
    coalescence_coeff = 1 / 3.14 / 4 / 100
    kernel_func = x -> coalescence_coeff
    kernel = CoalescenceTensor(kernel_func, 0, FT(100))
    ODE_parameters = Dict(
        :dist => [dist_init],
        :kernel => settings.kernel,
        :dt => FT(1)
        )
    #rhs = make_box_model_rhs(OneModeCoalStyle())
    rhs(m, par, t) = get_int_coalescence(OneModeCoalStyle(), m, par, par[:kernel])
    prob = ODEProblem(rhs, moments_init, settings.tspan, ODE_parameters)
    #sol = solve(prob, SSPRK33(), dt = ODE_parameters[:dt])
    sol = solve(prob, CVODE_BDF(), alg_hints = [:stiff], reltol = tol, abstol = tol)
    #
    ################ above is the new setup

    # Return moments at last time step
    moments_final = vcat(sol.u'...)[end, :]

    return moments_final
end

"""
  make_box_model_rhs(coal_type::CoalescenceStyle)

  `coal_type` type of coal source term function: OneModeCoalStyle, TwoModesCoalStyle, NumericalCoalStyle
Returns a function representing the right hand side of the ODE equation containing divergence 
of coalescence source term.
"""
# TODO: update the analytical coalescence style to NOT re-allocate matrices
function make_box_model_rhs(coal_type::AnalyticalCoalStyle)
    rhs(m, par, t) = get_int_coalescence(coal_type, m, par, par[:kernel])
end

end # module
