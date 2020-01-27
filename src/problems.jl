"""
    CESProblem

An uncertainty quantification problem.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct CESProblem{P,M,O,S<:HilbertSpace}
    "The prior distribution for the parameters ``θ``. Currently only `MvNormal` objects are supported."
    prior::P

    "A forward model maps the parameter ``θ`` to a predicted observation."
    forwardmodel::M

    "The observed data."
    obs::O

    "The space on which the forward model output and observation exist. [`DefaultSpace`](@ref) is the default."
    space::S
end
CESProblem(prior, forwardmodel, obs) = CESProblem(prior, forwardmodel, obs, DefaultSpace())


struct FlatPrior
end

Distributions.logpdf(::FlatPrior, x) = 0.0

function neglogposteriordensity(s::CESProblem, θ)
    -logpdf(s.prior, θ) + norm(s.forwardmodel(θ) - s.obs, s.space)
end

