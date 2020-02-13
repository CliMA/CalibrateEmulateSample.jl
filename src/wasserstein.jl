#####
##### Wasserstein distance
#####

function pysearchsorted(a,b;side="left")
    if side == "left"
        return searchsortedfirst.(Ref(a),b) .- 1
    else
        return searchsortedlast.(Ref(a),b)
    end
end

function _cdf_distance(p, u_values, v_values, u_weights=nothing, v_weights=nothing)
    _validate_distribution!(u_values, u_weights)
    _validate_distribution!(v_values, v_weights)

    u_sorter = sortperm(u_values)
    v_sorter = sortperm(v_values)

    all_values = vcat(u_values, v_values)
    sort!(all_values)

    # Compute the differences between pairs of successive values of u and v.
    deltas = diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = pysearchsorted(u_values[u_sorter],all_values[1:end-1], side="right")
    v_cdf_indices = pysearchsorted(v_values[v_sorter],all_values[1:end-1], side="right")

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights == nothing
        u_cdf = (u_cdf_indices) / length(u_values)
    else
        u_sorted_cumweights = vcat([0], cumsum(u_weights[u_sorter]))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[end]
    end

    if v_weights == nothing
        v_cdf = (v_cdf_indices) / length(v_values)
    else
        v_sorted_cumweights = vcat([0], cumsum(v_weights[v_sorter]))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[end]
    end

    # Compute the value of the integral based on the CDFs.
    if p == 1
        return sum(abs.(u_cdf - v_cdf) .* deltas)
    end
    if p == 2
        return sqrt(sum((u_cdf - v_cdf).^2 .* deltas))
    end
    return sum(abs.(u_cdf - v_cdf).^p .* deltas)^(1/p)
end

function _validate_distribution!(vals, weights)
    # Validate the value array.
    length(vals) == 0 && throw(ValueError("Distribution can't be empty."))
    # Validate the weight array, if specified.
    if weights ≠ nothing
        if length(weights) != length(vals)
            throw(ValueError("Value and weight array-likes for the same
                              empirical distribution must be of the same size."))
        end
        any(weights .< 0) && throw(ValueError("All weights must be non-negative."))
        if !(0 < sum(weights) < Inf)
            throw(ValueError("Weight array-like sum must be positive and
                              finite. Set as None for an equal distribution of
                              weight."))
        end
    end
    return nothing
end

function wasserstein_distance(u_values, v_values, u_weights=nothing, v_weights=nothing)
    return _cdf_distance(1, u_values, v_values, u_weights, v_weights)
end
