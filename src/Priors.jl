module Priors

using Distributions
using DocStringExtensions
using Random

export Prior
export Deterministic


"""
    Deterministic(value::FT)

A deterministic distribution is a distribution over only one value, which it takes with probability 1.
It can be used when a distribution is required, but the outcome is deterministic. 
```
d = Deterministic(3.0)  # Deterministic "distribution" with value 3.0
rand(d)                 # Always returns the same value
```
"""
struct Deterministic{FT<:Real} <: DiscreteUnivariateDistribution
    value::FT
end

eltype(::Deterministic{FT}) where {FT} = FT
rand(rng::AbstractRNG, d::Deterministic) = d.value
pdf(d::Deterministic, x::Real) = x == d.value ? 1.0 : 0.0
cdf(d::Deterministic, x::Real) = x < d.value ? 0.0 : 1.0
quantile(d::Deterministic{FT}, p) where {FT} = 0 <= p <= 1 ? d.value : FT(NaN)
minimum(d::Deterministic) = d.value
maximum(d::Deterministic) = d.value
insupport(d::Deterministic, x) = x == d.value
mean(d::Deterministic) = d.value
var(d::Deterministic) = 0.0
mode(d::Deterministic) = d.value
skewness(d::Deterministic) = 0.0
kurtosis(d::Deterministic) = 0.0
entropy(d::Deterministic) = 0.0
mgf(d::Deterministic, t) = exp(t * d.value)
cf(d::Deterministic, t) = exp(im * t * d.value)


"""
    struct Prior

Structure to hold information on the prior distribution of a UQ parameter

# Fields
$(DocStringExtensions.FIELDS)
"""
struct Prior
    "prior distribution"
    dist::Union{Distribution, Deterministic}
    "name of the parameter with prior distribution dist"
    param_name::String
end


"""
    Prior(value::FT, param_name::String)
"""
function Prior(value::FT, param_name::String) where {FT<:Real}
    Prior(Deterministic(value), param_name)
end


end # module 
