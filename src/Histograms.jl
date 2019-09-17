module Histograms
"""
This module is mostly a convenient wrapper of Python functions (numpy, scipy).

Functions in this module:
 - W1 (2 methods)

"""

import PyCall
import NPZ
include("ConvenienceFunctions.jl")

scsta = PyCall.pyimport("scipy.stats")

################################################################################
# HistData struct ##############################################################
################################################################################
mutable struct HistData
  samples::Dict{Symbol, AbstractVecOrMat}
end

HistData() = HistData(Dict())

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

function load!(hd::HistData, key::Symbol, filename::String)
  samples = NPZ.npzread(filename)
  if isa(samples, Array)
    load!(hd, key, samples)
  else
    throw(error("load!: ", filename, " is not an Array; abort"))
  end
end

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

plot(hd::HistData, plt, key::Symbol; kws...) =
  plot(hd, plt, key, UnitRange(1, size(hd.samples[key], 1)); kws...)

function plot_raw(S::Vector, plt; kws...)
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
                     normalize = true)
  L = maximum([u_samples; v_samples]) - minimum([u_samples; v_samples])
  return if !normalize
    scsta.wasserstein_distance(u_samples, v_samples)
  else
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

`U_samples` and `V_samples` should have samples in the 2nd dimension (along
rows) and have the same 1st dimension (same number of rows). If not, the minimum
of the two (minimum number of rows) will be taken.

`normalize` induces *pairwise* normalization, i.e. it max's and min's are
computed for each pair (u_j, v_j) individually.

Returns:
  - w1_UV:         array-like; the pairwise Wasserstein-1 distances:
                   w1(u1, v1)
                   w1(u2, v2)
                   ...
                   w1(u_K, v_K)
"""
function W1(U_samples::AbstractMatrix, V_samples::AbstractMatrix;
                     normalize = true)
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

function W1(hd::HistData, key1::Symbol, key2::Symbol, k::Union{Int,UnitRange})
  if isa(hd.samples[key1], Vector)
    u = hd.samples[key1]
  else
    u = vec(hd.samples[key1][k, 1:end])
  end
  if isa(hd.samples[key2], Vector)
    v = hd.samples[key2]
  else
    v = vec(hd.samples[key2][k, 1:end])
  end
  return W1(u, v)
end

function W1(hd::HistData, key::Symbol)
  key2all_vec = Dict{Symbol, Vector{Float64}}()
  for ki in keys(hd.samples)
    if ki == key
      continue
    end
    key2all_vec[ki] = W1(hd, ki, key)
  end
  return key2all_vec
end

function W1(hd::HistData, key1::Symbol, key2::Symbol)
  return W1(hd.samples[key1], hd.samples[key2])
end

end # module


