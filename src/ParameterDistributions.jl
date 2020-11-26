module ParameterDistributionsStorage

## Imports
using Distributions
using StatsBase
using Random

## Exports

#objects
export Parameterized, Samples
export NoConstraint, BoundedBelow, BoundedAbove, Bounded
export ParameterDistribution, ParameterDistributions

#functions
export get_name, get_distribution
export sample_distribution
export transform_real_to_prior, transform_prior_to_real

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
    Samples{FT<:Real} <: ParameterDistributionType

A distribution comprised of only samples
"""
struct Samples{FT<:Real} <: ParameterDistributionType
    distribution_samples::Array{FT}
end 


# For the transforms
abstract type ConstraintType end

"""
    NoConstraint <: ConstraintType

No constraint.
"""

struct NoConstraint <: ConstraintType end

"""
    BoundedBelow{FT <: Real} <: ConstraintType

A lower bound constraint.
"""
struct BoundedBelow{FT <: Real} <: ConstraintType
    lower_bound::FT
end
"""
    BoundedAbove{FT <: Real} <: ConstraintType

And upper bound constraint
"""
struct BoundedAbove{FT <: Real} <: ConstraintType
    upper_bound::FT
end

"""
    Bounded{FT <: Real} <: ConstraintType

Both a lower and upper bound constraint.
"""
struct Bounded{FT <: Real} <: ConstraintType
    lower_bound::FT
    upper_bound::FT
    Bounded(lower_bound::FT,upper_bound::FT) where {FT <: Real} = upper_bound <= lower_bound ? throw(DomainError("upper bound must be greater than lower bound")) : new{FT}(lower_bound, upper_bound)
end

"""
    len(c::Array{CType})

The number of constraints, each constraint has length 1.
"""
function len(c::CType) where {CType <: ConstraintType}
    return 1
end

function len(carray::Array{CType}) where {CType <: ConstraintType}
    return size(carray)[1]
end
"""
    len(d::ParametrizedDistributionType)

The number of dimensions of the parameter space
"""
function len(d::Parameterized) 
    return length(d.distribution)
end

function len(d::Samples)
    return size(d.distribution_samples)[2]
end

"""
    struct ParameterDistribution

Structure to hold a parameter distribution
"""
struct ParameterDistribution{PDType <: ParameterDistributionType, CType <: ConstraintType}
    distribution::PDType
    constraints::Array{CType}
    name::String

    ParameterDistribution(parameter_distribution::PDType,
                          constraints::Array{CType},
                          name::String) where {PDType <: ParameterDistributionType,
                                               CType <: ConstraintType} = !(len(parameter_distribution) ==
len(constraints)) ? throw(DimensionMismatch("There must be one constraint per parameter")) : new{PDType,CType}(parameter_distribution, constraints, name)     
end
    

"""
    struct ParameterDistributions

Structure to hold an array of ParameterDistribution's
"""
struct ParameterDistributions
    parameter_distributions::Array{ParameterDistribution}
end




## Functions

"""
    get_name(pds::ParameterDistributions)

Returns a list of ParameterDistribution names
"""
function get_name(pds::ParameterDistributions)
    return [get_name(pd) for pd in pds.parameter_distributions]
end

function get_name(pd::ParameterDistribution)
    return pd.name
end

"""
    get_distributions(pds::ParameterDistributions)

Returns a `Dict` of `ParameterDistribution` distributions by name, (unless sample type)
"""
function get_distribution(pds::ParameterDistributions)
    return Dict{String,Any}(get_name(pd) => get_distribution(pd) for pd in pds.parameter_distributions)
end

function get_distribution(pd::ParameterDistribution)
    return get_distribution(pd.distribution)
end

function get_distribution(d::Samples)
    return "Contains samples only"
end
function get_distribution(d::Parameterized)
    return d.distribution
end

"""
    function sample_distribution(pds::ParameterDistributions)

Draws samples from the parameter distributions
"""
function sample_distribution(pds::ParameterDistributions)
    return sample_distribution(pds,1)
end

function sample_distribution(pds::ParameterDistributions, n_samples::IT) where {IT <: Integer}
    return Dict{String,Any}(get_name(pd) => sample_distribution(pd, n_samples) for pd in pds.parameter_distributions)
end

function sample_distribution(pd::ParameterDistribution)
    return sample_distribution(pd,1)
end

function sample_distribution(pd::ParameterDistribution, n_samples::IT) where {IT <: Integer}
    return sample_distribution(pd.distribution,n_samples)
end

function sample_distribution(d::Samples, n_samples::IT)  where {IT <: Integer}
    total_samples = size(d.distribution_samples)[1]
    samples_idx = StatsBase.sample(collect(1:total_samples), n_samples; replace=false)
    return d.distribution_samples[samples_idx, :]
end

function sample_distribution(d::Parameterized,n_samples::IT) where {IT <: Integer}
    return rand(d.distribution, n_samples)
end


#apply transforms

"""
    transform_real_to_prior(pds::ParameterDistributions, x::Array{Real})

Apply the transformation to map real (and possibly constrained) parameters `xarray` into the unbounded prior space
"""
function transform_real_to_prior(pds::ParameterDistributions, xarray::Array{FT}) where {FT <: Real}
    #split xarray into chunks to be fed into each distribution
    clen = [len(pd.constraints) for pd in pds.parameter_distributions] # e.g [2,3,1,3]
    cumulative_clen = [sum(clen[1:i]) for i = 1:size(clen)[1]] # e.g [1 1:2, 3:5, 6, 7:9 ]
    x_idx = Dict{Integer,Array{Integer}}(i => collect(cumulative_clen[i - 1] + 1:cumulative_clen[i])
                                         for i in 2:size(cumulative_clen)[1])
    x_idx[1] = collect(1:cumulative_clen[1])

    return cat([transform_real_to_prior(pd,xarray[x_idx[i]]) for (i,pd) in enumerate(pds.parameter_distributions)]...,dims=1)
end

function transform_real_to_prior(pd::ParameterDistribution, xarray::Array{FT}) where {FT <: Real}
    return [transform_real_to_prior(constraint,xarray[i]) for (i,constraint) in enumerate(pd.constraints)]
end

"""
No constraint mapping x -> x
"""
function transform_real_to_prior(c::NoConstraint , x::FT) where {FT <: Real}
    return x
end

"""
Bounded below -> unbounded, use mapping x -> log(x - lower_bound)
"""
function transform_real_to_prior(c::BoundedBelow, x::FT) where {FT <: Real}    
    return log(x - c.lower_bound)
end

"""
Bounded above -> unbounded, use mapping x -> log(upper_bound - x)
"""
function transform_real_to_prior(c::BoundedAbove, x::FT) where {FT <: Real}    
    return log(c.upper_bound - x)
end

"""
Bounded -> unbounded, use mapping x -> log((x - lower_bound) / (upper_bound - x)
"""
function transform_real_to_prior(c::Bounded, x::FT) where {FT <: Real}    
    return log( (x - c.lower_bound) / (c.upper_bound - x))
end



"""
    transform_prior_to_real(pds::ParameterDistributions, xarray::Array{Real})

Apply the transformation to map parameters `xarray` from the unbounded space into (possibly constrained) real space
"""
function transform_prior_to_real(pds::ParameterDistributions,
                                 xarray::Array{FT}) where {FT <: Real}

    #split xarray into chunks to be fed into each distribution
    clen = [len(pd.constraints) for pd in pds.parameter_distributions] # e.g [2,3,1,3]
    cumulative_clen = [sum(clen[1:i]) for i = 1:size(clen)[1]] # e.g [1 1:2, 3:5, 6, 7:9 ]
    x_idx = Dict{Integer,Array{Integer}}(i => collect(cumulative_clen[i-1]+1:cumulative_clen[i])
                                         for i in 2:size(cumulative_clen)[1])
    x_idx[1] = collect(1:cumulative_clen[1])
    
    return cat([transform_prior_to_real(pd,xarray[x_idx[i]]) for (i,pd) in enumerate(pds.parameter_distributions)]...,dims=1)
end

function transform_prior_to_real(pd::ParameterDistribution, xarray::Array{FT}) where {FT <: Real}
    return [transform_prior_to_real(constraint,xarray[i]) for (i,constraint) in enumerate(pd.constraints)]
end

"""
No constraint mapping x -> x
"""
function transform_prior_to_real(c::NoConstraint , x::FT) where {FT <: Real}
    return x
end

"""
Unbounded -> bounded below, use mapping x -> exp(x) + lower_bound
"""
function transform_prior_to_real(c::BoundedBelow, x::FT) where {FT <: Real}    
    return exp(x) + c.lower_bound
end

"""
Unbounded -> bounded above, use mapping x -> upper_bound - exp(x)
"""
function transform_prior_to_real(c::BoundedAbove, x::FT) where {FT <: Real}    
    return c.upper_bound - exp(x)
end

"""
Unbounded -> bounded, use mapping x -> (upper_bound * exp(x) + lower_bound) / (exp(x) + 1)
"""
function transform_prior_to_real(c::Bounded, x::FT) where {FT <: Real}    
    return (c.upper_bound * exp(x) + c.lower_bound) / (exp(x) + 1)
end



end # of module
