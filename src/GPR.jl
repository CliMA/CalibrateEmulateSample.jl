module GPR

using Parameters # lets you have defaults for fields

using EllipsisNotation # adds '..' to refer to the rest of array
import ScikitLearn
import StatsBase
const sklearn = ScikitLearn

sklearn.@sk_import gaussian_process : GaussianProcessRegressor
sklearn.@sk_import gaussian_process.kernels : (RBF, Matern, WhiteKernel)


"""
A simple struct to handle Gaussian Process Regression related stuff

Functions that operate on GPR.Wrap struct:
 - set_data! (1 method)
 - subsample! (3 methods)
 - learn! (1 method)
 - predict (1 method)

Do *not* set Wrap's variables except for `thrsh`; use setter functions!
"""
@with_kw mutable struct Wrap
  thrsh::Int = 500
  data = nothing
  subsample = nothing
  GPR = nothing
  __data_set::Bool = false
  __subsample_set::Bool = false
end

################################################################################
# GRPWrap-related functions ####################################################
################################################################################
"""
Set `gprw.data` and reset `gprw.subsample` and `gprw.GPR` -- very important!

Parameters:
  - gprw:        an instance of GPR.Wrap
  - data:        input data to learn from (at least 2-dimensional)

`data` should be in the following format:
  last column: values/labels/y values
  first column(s): locations/x values
"""
function set_data!(gprw::Wrap, data::Array{<:Real})
  if ndims(data) > 2
    println(warn("set_data!"), "ndims(data) > 2; will use the first two dims")
    idx = fill(1, ndims(data) - 2)
    data = data[:,:,idx...]
  elseif ndims(data) < 2
    throw(error("set_data!: ndims(data) < 2; cannot proceed"))
  end
  gprw.data = data
  gprw.subsample = nothing
  gprw.GPR = nothing
  gprw.__data_set = true
  gprw.__subsample_set = false
  println(name("set_data!"), size(gprw.data,1), " points")
  flush(stdout)
end

"""
Subsample `gprw.data` using `indices`

Parameters:
  - gprw:        an instance of GPR.Wrap
  - indices:     indices that will be used to subsample `gprw.data`
"""
function subsample!(gprw::Wrap; indices::Union{Array{Int,1}, UnitRange{Int}})
  gprw.subsample = @view(gprw.data[indices,..])
  gprw.__subsample_set = true
  println(name("subsample!"), size(gprw.subsample,1), " subsampled")
  flush(stdout)
end

"""
Draw `thrsh` subsamples from `gprw.data`

Parameters:
  - gprw:        an instance of GPR.Wrap
  - thrsh:       threshold for the maximum number of points used in subsampling

If `thrsh` > 0 and `thrsh` < number of `gprw.data` points:
  subsample `thrsh` points uniformly randomly from `gprw.data`
If `thrsh` > 0 and `thrsh` >= number of `gprw.data` points:
  no subsampling, use whole `gprw.data`
If `thrsh` < 0:
  no subsampling, use whole `gprw.data`

This function ignores `gprw.thrsh`
"""
function subsample!(gprw::Wrap, thrsh::Int)
  if !gprw.__data_set
    throw(error("subsample!: 'data' is not set, cannot sample"))
  end
  if thrsh == 0
    throw(error("subsample!: 'thrsh' == 0, cannot sample"))
  end

  N = size(gprw.data,1)
  if thrsh < 0
    thrsh = N
  end

  if N > thrsh
    inds = StatsBase.sample(1:N, thrsh, replace = false)
  else
    inds = 1:N
  end

  subsample!(gprw, indices = inds)
end

"""
Wrapper for subsample!(gprw::Wrap, thrsh:Int)
"""
function subsample!(gprw::Wrap)
  subsample!(gprw, gprw.thrsh)
end

"""
Fit a GP regressor to `gprw.data` that was previously set

Parameters:
  - gprw:        an instance of GPR.Wrap
  - kernel:      "rbf" or "matern"; "rbf" by default
  - noise:       non-optimized noise level for the RBF kernel
                 (in addition to the optimized one)
  - nu:          Matern's nu parameter (smoothness of functions)
"""
function learn!(gprw::Wrap; kernel::String = "rbf", noise = 0.5, nu = 1.5)
  if !gprw.__subsample_set
    println(warn("learn!"), "'subsample' is not set; attempting to set...")
    subsample!(gprw)
  end

  WK = WhiteKernel(1, (1e-10, 10))
  if kernel == "matern"
    GPR_kernel = 1.0 * Matern(length_scale = 1.0, nu = nu) + WK
  else # including "rbf", which is the default
    if kernel != "rbf"
      println(warn("learn!"), "Kernel '", kernel, "' is not supported; ",
              "falling back to RBF")
    end
    GPR_kernel = 1.0 * RBF(1.0, (1e-10, 1e+6)) + WK
  end

  gprw.GPR = GaussianProcessRegressor(
      kernel = GPR_kernel,
      n_restarts_optimizer = 7,
      alpha = noise
      )
  sklearn.fit!(gprw.GPR, gprw.subsample[:,1:end-1], gprw.subsample[:,end])

  println(name("learn!"), gprw.GPR.kernel_)
  flush(stdout)
end

"""
Return mean (and st. deviation) values

Parameters:
  - gprw:        an instance of GPR.Wrap
  - x:           data for prediction
  - return_std:  boolean flag, whether to return st. deviation

Returns:
  - mean:        mean of the GP regressor at `x` locations
  - (mean, std): mean and st. deviation if `return_std` flag is true
"""
function predict(gprw::Wrap, x; return_std = false)
  if ndims(x) == 1
    # add an extra dimension to `x` if it's a vector (scikit-learn's whim)
    return gprw.GPR.predict(reshape(x, (size(x)...,1)), return_std = return_std)
  else
    return gprw.GPR.predict(x, return_std = return_std)
  end
end

################################################################################
# convenience functions ########################################################
################################################################################
const RPAD = 25

function name(name::AbstractString)
  return rpad(name * ":", RPAD)
end

function warn(name::AbstractString)
  return rpad("WARNING (" * name * "):", RPAD)
end

end # module


