module Transformations

export logmean_and_logstd
export logmean_and_logstd_from_mode_std
export mean_and_std_from_ln
export log_transform
export exp_transform

"""
    logmean_and_logstd(μ, σ)

Returns the lognormal parameters μ and σ from the mean μ and std σ of the 
lognormal distribution.
"""
function logmean_and_logstd(μ, σ)
    σ_log = sqrt(log(1.0 + σ^2/μ^2))
    μ_log = log(μ / (sqrt(1.0 + σ^2/μ^2)))
    return μ_log, σ_log
end

"""
    logmean_and_logstd_from_mode_std(μ, σ)

Returns the lognormal parameters μ and σ from the mode and the std σ of the 
lognormal distribution.
"""
function logmean_and_logstd_from_mode_std(mode, σ)
    σ_log = sqrt( log(mode)*(log(σ^2)-1.0)/(2.0-log(σ^2)*3.0/2.0) )
    μ_log = log(mode) * (1.0-log(σ^2)/2.0)/(2.0-log(σ^2)*3.0/2.0)
    return μ_log, σ_log
end

"""
    mean_and_std_from_ln(μ, σ)

Returns the mean and variance of the lognormal distribution
from the lognormal parameters μ and σ.
"""
function mean_and_std_from_ln(μ_log, σ_log)
    μ = exp(μ_log + σ_log^2/2)
    σ = sqrt( (exp(σ_log^2) - 1)* exp(2*μ_log + σ_log^2) )
    return μ, σ
end

log_transform(a::AbstractArray) = log.(a)
exp_transform(a::AbstractArray) = exp.(a)

end #module