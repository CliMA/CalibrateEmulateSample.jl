module ParameterDistributionStorage

## Imports
using Distributions
using StatsBase
using Random

## Exports

#objects
export Parameterized, Samples
export ParameterDistribution
export Constraint

#functions
export get_name, get_distribution
export sample_distribution
export no_constraint, bounded_below, bounded_above, bounded
export transform_constrained_to_unconstrained, transform_unconstrained_to_constrained

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

A distribution comprised of only samples, stored as columns of parameters
"""
struct Samples{FT<:Real} <: ParameterDistributionType
    distribution_samples::Array{FT,2} #parameters are columns
    Samples(distribution_samples::Array{FT,2}; params_are_columns=true) where {FT <: Real} = params_are_columns ? new{FT}(distribution_samples) : new{FT}(permutedims(distribution_samples,[2,1]))
    #Distinguish 1 sample of an ND parameter or N samples of 1D parameter, and store as 2D array  
    Samples(distribution_samples::Array{FT,1}; params_are_columns=true) where {FT <: Real} = params_are_columns ? new{FT}(reshape(distribution_samples,1,:)) : new{FT}(reshape(distribution_samples,:,1))
    
end 


# For the transforms
abstract type ConstraintType end
"""
    Constraint <: ConstraintType

Contains two functions to map between constrained and unconstrained spaces.
"""

struct Constraint <: ConstraintType
    constrained_to_unconstrained::Function
    unconstrained_to_constrained::Function
end


"""
    no_constraint()

Constructs a Constraint with no constraints, enforced by maps x -> x and x -> x.
"""
function no_constraint()
    c_to_u = (x -> x)
    u_to_c = (x -> x)
    return Constraint(c_to_u,u_to_c)
end
    
"""
    bounded_below(lower_bound::FT) where {FT <: Real}

Constructs a Constraint with provided lower bound, enforced by maps x -> log(x - lower_bound) and x -> exp(x) + lower_bound.
"""
function bounded_below(lower_bound::FT) where {FT <: Real}
    c_to_u = ( x -> log(x - lower_bound) )
    u_to_c = ( x -> exp(x) + lower_bound ) 
    return Constraint(c_to_u, u_to_c)
end

"""
    bounded_above(upper_bound::FT) where {FT <: Real} 

Constructs a Constraint with provided upper bound, enforced by maps x -> log(upper_bound - x) and x -> upper_bound - exp(x).
"""
function bounded_above(upper_bound::FT) where {FT<:Real}
    c_to_u = ( x -> log(upper_bound - x) )
    u_to_c = ( x -> upper_bound - exp(x) ) 
    return Constraint(c_to_u, u_to_c)
end
    

"""
    Bounded{FT <: Real} <: ConstraintType

Constructs a Constraint with provided upper and lower bounds, enforced by maps
x -> log((x - lower_bound) / (upper_bound - x))
and
x -> (upper_bound * exp(x) + lower_bound) / (exp(x) + 1)

"""
function bounded(lower_bound::FT, upper_bound::FT) where {FT <: Real}
    if (upper_bound <= lower_bound)
        throw(DomainError("upper bound must be greater than lower bound"))
    end
    c_to_u = ( x -> log( (x - lower_bound) / (upper_bound - x)) )
    u_to_c = ( x -> (upper_bound * exp(x) + lower_bound) / (exp(x) + 1) )
    return Constraint(c_to_u, u_to_c)
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
    dimension(d<:ParametrizedDistributionType)

The number of dimensions of the parameter space
"""
function dimension(d::Parameterized) 
    return length(d.distribution)
end

function dimension(d::Samples)
    return size(d.distribution_samples)[1]
end

"""
    n_samples(d::Samples)

The number of samples in the array
"""
function n_samples(d::Samples)
    return size(d.distribution_samples)[2]
end

"""
    struct ParameterDistribution

Structure to hold a parameter distribution, always stored as an array of distributions
"""
struct ParameterDistribution{PDType <: ParameterDistributionType, CType <: ConstraintType, ST <: AbstractString}
    distributions::Array{PDType}
    constraints::Array{CType}
    names::Array{ST}

    function ParameterDistribution(parameter_distributions::Union{PDType,Array{PDType}},
                                   constraints::Union{CType, Array{CType}, Array},
                                   names::Union{ST, Array{ST}}) where {PDType <: ParameterDistributionType,
                                                                       CType <: ConstraintType,
                                                                       ST <: AbstractString}
        
        parameter_distributions = isa(parameter_distributions, PDType) ? [parameter_distributions] : parameter_distributions
        constraints = isa(constraints, Union{<:ConstraintType,Array{<:ConstraintType}}) ? [constraints] : constraints #to calc n_constraints_per_dist
        names = isa(names, ST) ? [names] : names
            
        n_parameter_per_dist = [dimension(pd) for pd in parameter_distributions]
        n_constraints_per_dist = [len(c) for c in constraints]
        n_dists = length(parameter_distributions)
        n_names = length(names)        
        if !(n_parameter_per_dist == n_constraints_per_dist)
            throw(DimensionMismatch("There must be one constraint per parameter in a distribution, use NoConstraint() type if no constraint is required"))
        elseif !(n_dists == n_names)
            throw(DimensionMismatch("There must be one name per parameter distribution"))
        else            
            constraints = cat(constraints...,dims=1)
            
            new{PDType,ConstraintType,ST}(parameter_distributions, constraints, names)
        end
    end
    
end
    


## Functions

"""
    get_name(pd::ParameterDistribution)

Returns a list of ParameterDistribution names
"""
function get_name(pd::ParameterDistribution)
    return pd.names
end

"""
    get_distribution(pd::ParameterDistribution)

Returns a `Dict` of `ParameterDistribution` distributions by name, (unless sample type)
"""
function get_distribution(pd::ParameterDistribution)
    return Dict{String,Any}(pd.names[i] => get_distribution(d) for (i,d) in enumerate(pd.distributions))
end

function get_distribution(d::Samples)
    return "Contains samples only"
end
function get_distribution(d::Parameterized)
    return d.distribution
end

"""
    function sample_distribution(pd::ParameterDistribution)

Draws samples from the parameter distributions
"""
function sample_distribution(pd::ParameterDistribution)
    return sample_distribution(pd,1)
end

function sample_distribution(pd::ParameterDistribution, n_draws::IT) where {IT <: Integer}
    return cat([sample_distribution(d,n_draws) for d in pd.distributions]...,dims=1)
end

function sample_distribution(d::Samples, n_draws::IT)  where {IT <: Integer}
    n_stored_samples = n_samples(d)
    samples_idx = StatsBase.sample(collect(1:n_stored_samples), n_draws)
    return d.distribution_samples[:,samples_idx]

end

function sample_distribution(d::Parameterized, n_draws::IT) where {IT <: Integer}
    return rand(d.distribution, n_draws)
end


#apply transforms

"""
    transform_constrained_to_unconstrained(pd::ParameterDistribution, x::Array{Real})

Apply the transformation to map (possibly constrained) parameters `xarray` into the unconstrained space
"""
function transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{FT}) where {FT <: Real}
    #split xarray into chunks 
#    return cat([transform_constrained_to_unconstrained(c,xarray[i]) for (i,c) in enumerate(pd.constraints)]...,dims=1)
    return cat([c.constrained_to_unconstrained(xarray[i]) for (i,c) in enumerate(pd.constraints)]...,dims=1)
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{Real})

Apply the transformation to map parameters `xarray` from the unconstrained space into (possibly constrained) space
"""
function transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{FT}) where {FT <: Real}
#    return [transform_unconstrained_to_constrained(c,xarray[i]) for (i,c) in enumerate(pd.constraints)]
    return cat([c.unconstrained_to_constrained(xarray[i]) for (i,c) in enumerate(pd.constraints)]...,dims=1)
end




# """
# No constraint mapping x -> x
# """
# function transform_constrained_to_unconstrained(c::NoConstraint , x::FT) where {FT <: Real}
#     return x
# end

# """
# Bounded below -> unbounded, use mapping x -> log(x - lower_bound)
# """
# function transform_constrained_to_unconstrained(c::BoundedBelow, x::FT) where {FT <: Real}    
#     return log(x - c.lower_bound)
# end

# """
# Bounded above -> unbounded, use mapping x -> log(upper_bound - x)
# """
# function transform_constrained_to_unconstrained(c::BoundedAbove, x::FT) where {FT <: Real}    
#     return log(c.upper_bound - x)
# end

# """
# Bounded -> unbounded, use mapping x -> log((x - lower_bound) / (upper_bound - x)
# """
# function transform_constrained_to_unconstrained(c::Bounded, x::FT) where {FT <: Real}    
#     return log( (x - c.lower_bound) / (c.upper_bound - x))
# end




# """
# No constraint mapping x -> x
# """
# function transform_unconstrained_to_constrained(c::NoConstraint , x::FT) where {FT <: Real}
#     return x
# end

# """
# Unbounded -> bounded below, use mapping x -> exp(x) + lower_bound
# """
# function transform_unconstrained_to_constrained(c::BoundedBelow, x::FT) where {FT <: Real}    
#     return exp(x) + c.lower_bound
# end

# """
# Unbounded -> bounded above, use mapping x -> upper_bound - exp(x)
# """
# function transform_unconstrained_to_constrained(c::BoundedAbove, x::FT) where {FT <: Real}    
#     return c.upper_bound - exp(x)
# end

# """
# Unbounded -> bounded, use mapping x -> (upper_bound * exp(x) + lower_bound) / (exp(x) + 1)
# """
# function transform_unconstrained_to_constrained(c::Bounded, x::FT) where {FT <: Real}    
#     return (c.upper_bound * exp(x) + c.lower_bound) / (exp(x) + 1)
# end



end # of module
