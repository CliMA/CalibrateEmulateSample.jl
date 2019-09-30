module Histograms
"""
This module is mostly a convenient wrapper of Python functions (numpy, scipy).

Functions in this module:
 - W1 (8 methods)
 - load! (3 methods)
 - plot (2 methods)
 - plot_raw (1 method)

"""

import PyCall
import NPZ
include("ConvenienceFunctions.jl")

scsta = PyCall.pyimport("scipy.stats")

################################################################################
# HistData struct ##############################################################
################################################################################
"""
A simple struct to store samples for empirical PDFs (histograms, distances etc.)

Functions that operate on HistData struct:
 - W1 (4 methods)
 - load! (3 methods)
 - plot (2 methods)
 - plot_raw (1 method)
"""
mutable struct HistData
  samples::Dict{Symbol, AbstractVecOrMat}
end

HistData() = HistData(Dict())

"""
Load samples into a HistData object under a specific key from a vector

Parameters:
  - hd:            HistData; groups of samples accessed by keys
  - key:           Symbol; key of the samples group to load samples to
  - samples:       array-like; samples to load

If the `key` group already exists, the samples are appended. If, in addition,
the samples were in a matrix, they are flattened first, and then the new ones
are added.
"""
function load!(hd::HistData, key::Symbol, samples::AbstractVector)
  if haskey(hd.samples, key)
    if isa(hd.samples[key], Matrix)
      println(warn("load!"),
              "hd.samples[:", key, "] is a Matrix; using vec(hd.samples)")
    end
    hd.samples[key] = vcat(vec(hd.samples[key]), samples)
  else
    hd.samples[key] = samples
  end
end

"""
Load samples into a HistData object under a specific key from a matrix

Parameters:
  - hd:            HistData; groups of samples accessed by keys
  - key:           Symbol; key of the samples group to load samples to
  - samples:       matrix-like; samples to load

If the `key` group already exists, the samples are appended. If, in addition,
the samples were in a vector, the new ones are flattened first, and then added
to the old ones. If the samples were in a matrix and row dimensions don't match,
the minimum of the two dimensions is chosen, the rest is discarded.
"""
function load!(hd::HistData, key::Symbol, samples::AbstractMatrix)
  if haskey(hd.samples, key)
    if isa(hd.samples[key], Vector)
      println(warn("load!"),
              "hd.samples[:", key, "] is a Vector; using vec(samples)")
      hd.samples[key] = vcat(hd.samples[key], vec(samples))
    elseif size(hd.samples[key], 1) != size(samples, 1)
      println(warn("load!"), "sizes of hd.samples & samples don't match; ",
              "squishing down to match the minimum of the two")
      K = min(size(hd.samples[key], 1), size(samples,1))
      hd.samples[key] = hcat(hd.samples[key][1:K, 1:end], samples[1:K, 1:end])
    else
      hd.samples[key] = hcat(hd.samples[key], samples)
    end
  else
    hd.samples[key] = samples
  end
end

"""
Load samples into a HistData object under a specific key from a file

Parameters:
  - hd:            HistData; groups of samples accessed by keys
  - key:           Symbol; key of the samples group to load samples to
  - filename:      String; name of a .npy file with samples (vector/matrix)

All the rules about vector/matrix interaction from two other methods apply.
"""
function load!(hd::HistData, key::Symbol, filename::String)
  samples = NPZ.npzread(filename)
  if isa(samples, Array) && ndims(samples) <= 2
    load!(hd, key, samples)
  else
    throw(error("load!: ", filename, " is not a 1- or 2-d Array; abort"))
  end
end

"""
Plot a histogram of a range of data by a specific key

Parameters:
  - hd:     HistData; groups of samples accessed by keys
  - plt:    a module used for plotting (only PyPlot supported)
  - key:    Symbol; key of the samples group to construct histogram from
  - k:      Int or UnitRange; if samples are in a matrix, which rows to use
  - kws:    dictionary-like; keyword arguments to pass to plotting function

If `hd.samples[key]` is a matrix, all the samples from `k` row(s) are combined;
if it is a vector, `k` is ignored.
"""
function plot(hd::HistData, plt, key::Symbol, k::Union{Int,UnitRange}; kws...)
  if length(hd.samples) == 0
    println(warn("plot"), "no samples, nothing to plot")
    return
  end
  is_pyplot = (Symbol(plt) == :PyPlot)
  if !is_pyplot
    println(warn("plot"), "only PyPlot is supported; not plotting")
    return
  end

  if isa(hd.samples[key], Vector)
    S = hd.samples[key]
  else
    S = vec(hd.samples[key][k, 1:end])
  end

  plot_raw(S, plt; kws...)
end

"""
Plot a histogram of whole data by a specific key

Parameters:
  - hd:     HistData; groups of samples accessed by keys
  - plt:    a module used for plotting (only PyPlot supported)
  - key:    Symbol; key of the samples group to construct histogram from
  - kws:    dictionary-like; keyword arguments to pass to plotting function
"""
plot(hd::HistData, plt, key::Symbol; kws...) =
  plot(hd, plt, key, UnitRange(1, size(hd.samples[key], 1)); kws...)

"""
Plot a histogram of samples

Parameters:
  - S:      array-like; samples to construct histogram from
  - plt:    a module used for plotting (only PyPlot supported)
  - kws:    dictionary-like; keyword arguments to pass to plotting function

The keyword arguments `kws` have precedence over defaults, but if left
unspecified, these are the defaults:
  bins     = "auto"
  histtype = "step"
  density  = true
"""
function plot_raw(S::AbstractVector, plt; kws...)
  kwargs_local = Dict{Symbol, Any}()
  kwargs_local[:bins]     = "auto"
  kwargs_local[:histtype] = "step"
  kwargs_local[:density]  = true

  # the order of merge is important! kws have higher priority
  plt.hist(S; merge(kwargs_local, kws)...)
end

################################################################################
# distance functions ###########################################################
################################################################################
"""
Compute the Wasserstein-1 distance between two distributions from their samples

Parameters:
  - u_samples:     array-like; samples from the 1st distribution
  - v_samples:     array-like; samples from the 2nd distribution
  - normalize:     boolean; whether to normalize the distance by 1/(max-min)

Returns:
  - w1_uv:         number; the Wasserstein-1 distance
"""
function W1(u_samples::AbstractVector, v_samples::AbstractVector;
                     normalize = false)
  return if !normalize
    scsta.wasserstein_distance(u_samples, v_samples)
  else
    u_m, u_M = extrema(u_samples)
    v_m, v_M = extrema(v_samples)
    L = max(u_M, v_M) - min(u_m, v_m)
    scsta.wasserstein_distance(u_samples, v_samples) / L
  end
end

"""
Compute the pairwise Wasserstein-1 distances between two sets of distributions
from their samples

Parameters:
  - U_samples:     matrix-like; samples from distributions (u1, u2, ...)
  - V_samples:     matrix-like; samples from distributions (v1, v2, ...)
  - normalize:     boolean; whether to normalize the distances by 1/(max-min)

Returns:
  - w1_UV:         array-like; the pairwise Wasserstein-1 distances:
                   w1(u1, v1)
                   w1(u2, v2)
                   ...
                   w1(u_K, v_K)

`U_samples` and `V_samples` should have samples in the 2nd dimension (along
rows) and have the same 1st dimension (same number of rows). If not, the minimum
of the two (minimum number of rows) will be taken.

`normalize` induces *pairwise* normalization, i.e. it max's and min's are
computed for each pair (u_j, v_j) individually.
"""
function W1(U_samples::AbstractMatrix, V_samples::AbstractMatrix;
                     normalize = false)
  if size(U_samples, 1) != size(V_samples, 1)
    println(warn("W1"), "sizes of U_samples & V_samples don't match; ",
            "will use the minimum of the two")
  end
  K = min(size(U_samples, 1), size(V_samples, 1))
  w1_UV = zeros(K)
  U_sorted = sort(U_samples[1:K, 1:end], dims = 2)
  V_sorted = sort(V_samples[1:K, 1:end], dims = 2)
  for k in 1:K
    w1_UV[k] = W1(U_sorted[k, 1:end], V_sorted[k, 1:end]; normalize = normalize)
  end
  return w1_UV
end

W1(U_samples::AbstractMatrix, v_samples::AbstractVector; normalize = false) =
  W1(vec(U_samples), v_samples; normalize = normalize)

W1(u_samples::AbstractVector, V_samples::AbstractMatrix; normalize = false) =
  W1(u_samples, vec(V_samples); normalize = normalize)

"""
Compute pairs of Wasserstein-1 distances between `key` samples and the rest

Parameters:
  - hd:     HistData; groups of samples accessed by keys
  - key:    Symbol; key of the samples group to compare everything else against
  - k:      Int or UnitRange; if samples are in a matrix, which rows to use

Returns:
  - key2all_combined:     Dict{Symbol, Float64}; pairs of W1 distances

Compute the W1-distances between `hd.samples[key]` and all other `hd.samples`.
If any of the `hd.samples` is a matrix (not a vector) then `k` is used to access
rows of said matrix, and then samples from these rows are combined together.
For any of the `hd.samples` that is a vector, `k` is ignored.

This function is useful when you have one reference (empirical) distribution and
want to compare the rest against that "ground truth" distribution.

Examples:
  k2a_row1 = W1(hd, :dns, 1)
  K = size(hd.samples[:dns], 1)
  k2a_combined = W1(hd, :dns, 1:K)
  println(k2a_combined[:bal])
"""
function W1(hd::HistData, key::Symbol, k::Union{Int,UnitRange})
  key2all_combined = Dict{Symbol, Float64}()
  for ki in keys(hd.samples)
    if ki == key
      continue
    end
    key2all_combined[ki] = W1(hd, ki, key, k)
  end
  return key2all_combined
end

"""
Compute a pair of Wasserstein-1 distance between `key1` & `key2` samples

Parameters:
  - hd:     HistData; groups of samples accessed by keys
  - key1:   Symbol; key of the first samples group
  - key2:   Symbol; key of the second samples group
  - k:      Int or UnitRange; if samples are in a matrix, which rows to use

Returns:
  - w1_key1key2:     Float64; W1 distance

Compute the W1-distance between `hd.samples[key1]` and `hd.samples[key2]`.
If either of them is a matrix (not a vector) then `k` is used to access rows of
said matrix, and then samples from these rows are combined together.
For vectors, `k` is ignored.

Examples:
  w1_dns_bal = W1(hd, :dns, :bal, 1)
  K = size(hd.samples[:dns], 1)
  w1_dns_bal_combined = W1(hd, :dns, :bal, 1:K)
"""
function W1(hd::HistData, key1::Symbol, key2::Symbol, k::Union{Int,UnitRange})
  u = if isa(hd.samples[key1], Vector)
    hd.samples[key1]
  else
    vec(hd.samples[key1][k, 1:end])
  end
  v = if isa(hd.samples[key2], Vector)
    hd.samples[key2]
  else
    vec(hd.samples[key2][k, 1:end])
  end
  return W1(u, v)
end

"""
Compute vectors of Wasserstein-1 distances between `key` samples and the rest

Parameters:
  - hd:     HistData; groups of samples accessed by keys
  - key:    Symbol; key of the samples group to compare everything else against

Returns:
  - key2all_vectorized:   Dict{Symbol, Union{Vector{Float64}, Float64}};
                          either vectors or pairs of W1 distances

Compute the W1-distances between `hd.samples[key]` and all other `hd.samples`.
For each pair of samples (`key` and something else) where both groups of samples
are in a matrix, the returned value will be a vector (corresponding to rows of
the matrices); for each pair where at least one of the groups is a vector, the
returned value will be a Float64, and all samples from a matrix are combined.

This function is useful when you have one reference (empirical) distribution and
want to compare the rest against that "ground truth" distribution.

Examples:
  k2a_vectorized = W1(hd, :dns)
  println(k2a_vectorized[:onl] .> 0.01)
"""
function W1(hd::HistData, key::Symbol)
  key2all_vectorized = Dict{Symbol, Union{Vector{Float64}, Float64}}()
  for ki in keys(hd.samples)
    if ki == key
      continue
    end
    key2all_vectorized[ki] = W1(hd, ki, key)
  end
  return key2all_vectorized
end

"""
Compute a vector of Wasserstein-1 distances between `key1` & `key2` samples

Parameters:
  - hd:     HistData; groups of samples accessed by keys
  - key1:   Symbol; key of the first samples group
  - key2:   Symbol; key of the second samples group

Returns:
  - w1_key1key2:     Union{Vector{Float64}, Float64}; W1 distance

Compute the W1-distance between `hd.samples[key1]` and `hd.samples[key2]`.

If both are matrices, the returned value will be a vector (corresponding to rows
of the matrices); if at least one of them is a vector, the returned value will
be a Float64, and all samples from a matrix are combined.

Examples:
  w1_dns_bal = W1(hd, :dns, :bal)
  println(w1_dns_bal .> 0.01)
"""
W1(hd::HistData, key1::Symbol, key2::Symbol) =
  W1(hd.samples[key1], hd.samples[key2])

end # module


