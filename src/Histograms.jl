module Histograms
"""
This module is mostly a convenient wrapper of Python functions (numpy, scipy).

Functions in this module:
 - wasserstein (2 methods)

"""

import PyCall
scsta = PyCall.pyimport("scipy.stats")

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
  - w1:            number; the Wasserstein-1 distance
"""
function wasserstein(u_samples::AbstractVector, v_samples::AbstractVector;
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
  - w1:            array-like; the pairwise Wasserstein-1 distances:
                   w1(u1, v1)
                   w1(u2, v2)
                   ...
                   w1(u_K, v_K)
"""
function wasserstein(U_samples::AbstractMatrix, V_samples::AbstractMatrix;
                     normalize = true)
  if size(U_samples, 1) != size(V_samples, 1)
    println(warn("wasserstein"), "sizes of U_samples & V_samples don't match; ",
            "will use the minimum of the two")
  end
  K = min(size(U_samples, 1), size(V_samples, 1))
  w1 = zeros(K)
  U_sorted = sort(U_samples[1:K, 1:end], dims = 2)
  V_sorted = sort(V_samples[1:K, 1:end], dims = 2)
  for k in 1:K
    w1[k] = wasserstein(U_sorted[k, 1:end], V_sorted[k, 1:end];
                        normalize = normalize)
  end
  return w1
end

end # module


