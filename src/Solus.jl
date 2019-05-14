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
    obs::O

    """
    The space on which the forward model output and observation exist. `DefaultSpace()` is the default.
    """
    space::S
end
SolusProblem(prior, forwardmodel, obs) = SolusProblem(prior, forwardmodel, obs, DefaultSpace())


"""
    Ensemble

An ensemble of `inputs` and their corresponding `outputs` from the forward model.

# Fields

$(DocStringExtensions.FIELDS)
"""
struct Ensemble{I,O}
    """
    A matrix of inputs: each column corresponds to a vector of ``θ``s
    """
    inputs::Vector{I}
    """
    Result of forward model for each column of `inputs`.
    """
    outputs::Vector{O}
end

"""
    neki_iter(prob::SolusProblem, ens::Ensemble)

Peform an iteration of NEKI, returning a new `Ensemble` object.
"""
function neki_iter(prob::SolusProblem, ens::Ensemble)
    covθ = cov(ens.inputs)
    covθ += (tr(covθ)*1e-15)I
    
    m = mean(ens.outputs)

    CG = [dot(Gθk - m, Gθj - prob.obs, prob.space) for Gθj in ens.outputs, Gθk in ens.outputs]
    
    Δt = 0.1 / norm(CG) # use better constants  
    implicit = lu( I + Δt .* (covθ / cov(prob.prior)) ) # todo: incorporate means
    noise = MvNormal(covθ)
    
    inputs = map(enumerate(ens.inputs)) do (j, θj)
        X = sum(enumerate(ens.inputs)) do (k, θk)
            CG[k,j]*θk
        end
        rhs = θj .- Δt .* X
        (implicit \ rhs) .+ sqrt(Δt)*rand(noise)
    end
    
    outputs = map(prob.forwardmodel, inputs)
    Ensemble(inputs, outputs)
end
    

end # module
