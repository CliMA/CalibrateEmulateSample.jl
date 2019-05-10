module Solus

using Distributions, Statistics, LinearAlgebra, DocStringExtensions

include("spaces.jl")

"""
    SolusProblem

An uncertainty quantification problem.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct SolusProblem{P<:MvNormal,M,O,S<:HilbertSpace}
    """
    The prior distribution for the parameters ``θ``. Currently only `MvNormal` objects are supported.
    """
    prior::P

    """
    A forward model maps the parameter ``θ`` to a predicted observation.
    """
    forwardmodel::M
    
    """
    The observed data.
    """
    observation::O

    """
    The space on which the forward model output and observation exist. `DefaultSpace()` is the default.
    """
    space::S
end
SolusModel(prior, forwardmodel, observation) = SolusModel(prior, forwardmodel, observation, DefaultSpace())


"""
    Ensemble

An ensemble of `inputs` and their corresponding `outputs` from the forward model.
"""
struct Ensemble{T,O}
    inputs::Matrix{T} # on your theta space: each row is a vector of θ
    outputs::Vector{O} # computed from forward model
end

"""
    neki_iter(prob::SolusProblem, ens::Ensemble)

Peform an iteration of NEKI, returning a new `Ensemble` object.
"""
function neki_iter(prob::SolusProblem, ens::Ensemble)
    covθ = cov(ens.inputs)
    covθ += (tr(covθ)*1e-15)I
    
    m = mean(ens.outputs)
    CG = [dot(u-prob.observation, v-m, prob.space) for u in ens.outputs, v in ens.outputs] #compute mean-field matrix
    
    Δt = 0.1 / norm(CG)
    implicit = lu( I + 1 * Δt .* covθ * cov(prior)) # todo: incorporate means
    rhsv = CG*ens.inputs
    rhs = ens.inputs - rhsv*Δt
    
    inputs = (implicit \ rhs) + rand(MvNormal(Δt .* covθ))
    outputs = [prob.forwardmodel(θ) for θ in eachslice(ens.inputs,dims=2)]
    Ensemble(inputs, outputs)
end
    

end # module
