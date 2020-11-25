module ParameterDistribution

## Imports
using Distributions
using StatsBase
using Random

## Exports

#objects
export ParameterDistribution

#functions
export set_distribution, get_distribution, sample_distribution
export transform_real_to_prior, transform_prior_to_real
#export apply_units_to_real


## Objects
# for the Distribution
abstract type ParameterDistributionType end

"""
    Parameterized <: ParameterDistributionType

A distribution constructed from a parametrized formula (e.g Julia Distributions.jl)
"""
struct Parameterized <: ParameterDistributionType
    distribution::Distribution
end

"""
    Samples{FT<:AbstractFloat} <: ParameterDistributionType

A distribution comprised of only samples
"""
struct Samples{FT<:AbstractFloat} <: ParameterDistributionType
    distribution_samples::Array{FT}
end 

# """
#     KernelDensityEstimate <: ParameterDistributionType
# 
# A distribution comprised of a Kernel Density Estimate of samples
# """
# struct KernelDensityEstimate <: ParameterDistributionType
#     # Not implemented
# end

# For the transforms
abstract type ConstraintType end

"""
    NoConstrain <: ConstraintType

The mapping x -> x
"""

struct NoConstraint <: ConstraintType end

"""
    BoundedBelow{FT <: AbstractFloat} <: ConstraintType

The mapping x -> ln(x - lower_bound{FT}); from a space bounded below to an unbounded space.
"""
struct BoundedBelow{FT <: AbstractFloat} <:
    lower_bound::FT
end

"""
    BoundedAbove{FT <: AbstractFloat} <: ConstraintType

The mapping to x -> ln( upper_bound{FT} - x); from a space bounded above to an unbounded space.
"""
struct BoundedAbove{FT <: AbstractFloat}
    upper_bound::FT
end

"""
    Bounded{FT <: AbstractFloat} <: ConstraintType

The mapping x -> ln( (x - lower_bound) / (upper_bound - x) ); from a bounded space to an unbounded space.
"""
struct Bounded{FT <: AbstractFloat}
    lower_bound::FT
    upper_bound::FT
end

function Bounded(lower::FT, upper::FT) where {FT <: AbstractFloat}
    @assert lower_bound < upper_bound
    Bounded{FT}(lower_bound,upper_bound)
end

"""
    struct ParameterDistribution

Structure to hold a parameter distribution
"""
struct ParameterDistribution{FT<:AbstractFloat,
                             PDType <: ParameterDistributionType,
                             CType <: ConstraintType}
    distribution::PDType
    constraints::Array{CType}
    name::String
end

function ParameterDistribution(parameter_distribution::PDType,
                               constraints::Array{CType},
                               name::String) where {FT <: AbstractFloat,
                                                    PDType <: ParameterDistributionType,
                                                    CType <: ConstraintType}
    #assert that total parameter dimensions matches number of constraints
    length(parameter_distribution) == length(constraints) ||
        throw(DimensionMismatch("There must be one constraint per parameter")) 
    
    ParameterDistribution{FT,PDType,CType}(parameter_distributions, constraints, name)
end

"""
    struct ParameterDistributions

Structure to hold an array of ParameterDistribution's
"""
struct ParameterDistributions
    parameter_distribution::Array{ParameterDistribution}
end




## Functions

"""
    get_names(pds::ParameterDistributions)

Returns a list of ParameterDistribution names
"""
function get_name(pds::ParameterDistributions)
    return [get_name(pd) for pd in pds]
end

function get_name(pd::ParameterDistribution)
    return pd.name
end

"""
    get_distributions(pds::ParameterDistributions)

Returns a `Dict` of `ParameterDistribution` distributions by name, (unless sample type)
"""
function get_distribution(pds::ParameterDistributions)
    return Dict{String,Any}(get_name(pd) => get_distribution(pd) for pd in pds)
end

function get_distribution(pd::ParameterDistribution{FT,Samples,CType}) where {FT <: AbstractFloat,
                                                                              CType <: ConstraintType}
    return "Contains samples only"
end
function get_distribution(pd::ParameterDistribution{FT,Parameterized,CType}) where {FT <: AbstractFloat,
                                                                              CType <: ConstraintType}
    return pd.distribution.distribution
end

"""
    function sample_distribution(pds::ParameterDistributions)

Draws samples from the parameter distributions
"""
function sample_distribution(pds::ParameterDistributions)
    return sample_distribution(pds,1)
end

function sample_distribution(pds::ParameterDistributions, n_samples::IT) where {IT <: Integer}
    return Dict{String,Any}(get_name(pd) => sample_distribution(pd, n_samples) for pd in pds)
end

function sample_distribution(pd::ParameterDistribution)
    return sample_distribution(pd,1)
end

function sample_distribution(pd::ParameterDistribution, n_samples::IT) where {IT <: Integer}
    return sample_distribution(pd.distribution)
end

function sample_distribution(d::Samples, n_samples::IT)  where {IT <: Integer}
    total_samples = size(d.distribution_samples)[1]
    samples_idx = sample(collect(1:total_samples), n_samples ; replace=false)
    return d.distribution_samples[samples_idx,:]
end

function sample_distribution(d::Parameterized,n_samples::IT) 
    return rand(pd.distirbution.distribution, n_samples)
end


#apply transforms
function length(c::CType) where {CType <: ConstraintType}
    return 1
end
function length(carray::Array{CType}) where {CType <: ConstraintType}
    return size(carray)[1]
end

"""
    transform_real_to_prior(pds::ParameterDistributions, x::Array{AbstractFloat})

Apply the transformation to map real parameter x into the unbounded prior space
"""
function transform_real_to_prior(pds::ParameterDistributions, xarray::Array{FT}) where {FT <: AbstractFloat}
    #split xarray into chunks to be fed into each distribution
    clen = [length(pd.constraints) for pd in pds] # e.g [2,3,1,3]
    cumulative_clen = [sum(clen[1:i]) for i = 1:size(clen)[1]] # e.g [1 1:2, 3:5, 6, 7:9 ]
    x_idx = Dict(Integer,Array{Integer})(i => collect(cumulative_cl[i-1]+1:cumulative_cl[i])
                                         for i in 2:size(cumulative_clen)[1])
    x_idx[1] = collect(1:cumulative_clen[1])

    return Dict{String,Any}(get_name(pd) => transform_real_to_prior(pd,xarray[x_idx[i]]) for (i,pd) in enumerate(pds))
end

function transform_real_to_prior(pd::ParameterDistribution, xarray::Array{FT}) where {FT <: AbstractFloat}
    return [transform_real_to_prior(constraint,xarray[i]) for (i,constraint) in enumerate(pd.constraints)]
end

function transform_real_to_prior(c::NoConstraints , x::Array{FT}) where {FT <: AbstractFloat}
    return x
end
function transform_real_to_prior(c::BoundedBelow, x::Array{FT}) where {FT <: AbstractFloat}    
    return ln(x - c.lower_bound)
end
function transform_real_to_prior(c::BoundedAbove, x::Array{FT}) where {FT <: AbstractFloat}    
    return ln(c.upper_bound - x)
end

function transform_real_to_prior(c::Bounded, x::Array{FT}) where {FT <: AbstractFloat}    
    return ln((c.upper_bound - x) / (x - c.lower_bound))
end



"""
    transform_prior_to_real(pds::ParameterDistributions, xarray::Array{AbstractFloat})

Apply the transformation to map parameter x into the real space
"""
function transform_prior_to_real(pds::ParameterDistributions,
                                 xarray::Array{FT}) where {FT <: AbstractFloat}

    #split xarray into chunks to be fed into each distribution
    clen = [length(pd.constraints) for pd in pds] # e.g [2,3,1,3]
    cumulative_clen = [sum(clen[1:i]) for i = 1:size(clen)[1]] # e.g [1 1:2, 3:5, 6, 7:9 ]
    x_idx = Dict(Integer,Array{Integer})(i => collect(cumulative_cl[i-1]+1:cumulative_cl[i])
                                         for i in 2:size(cumulative_clen)[1])
    x_idx[1] = collect(1:cumulative_clen[1])
    
    return Dict{String,Any}(get_name(pd) => transform_prior_to_real(pd,xarray[x_idx[i]])
                                             for (i,pd) in enumerate(pds))
end

function transform_prior_to_real(pd::ParameterDistributions, xarray::Array{FT}) where {FT <: AbstractFloat}
    return [transform_prior_to_real(constraint,xarray[i]) for (i,constraint) in enumerate(pd.constraints)]
end


function transform_prior_to_real(c::NoConstraints , x::FT) where {FT <: AbstractFloat}
    return x
end
function transform_prior_to_real(c::BoundedBelow, x::FT) where {FT <: AbstractFloat}    
    return exp(x) + c.lower_bound
end
function transform_prior_to_real(c::BoundedAbove, x::FT) where {FT <: AbstractFloat}    
    return c.upper_bound - exp(x)
end

function transform_prior_to_real(c::Bounded, x::FT) where {FT <: AbstractFloat}    
    return (c.upper_bound * exp(x) + c.lower_bound) / (exp(x) + 1)
end



end # of module
