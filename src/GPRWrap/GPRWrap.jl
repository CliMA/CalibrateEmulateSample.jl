using Parameters # lets you have defaults for fields

using EllipsisNotation # adds '..' to refer to the rest of array
import ScikitLearn
import StatsBase
const sklearn = ScikitLearn

sklearn.@sk_import gaussian_process : GaussianProcessRegressor
sklearn.@sk_import gaussian_process.kernels : (RBF, Matern, WhiteKernel)


@with_kw mutable struct GPRWrap
  """
  A simple struct to handle Gaussian Process Regression related stuff

  """
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
function set_data!(gprw::GPRWrap, data::Array{<:Real})
  """
  Set `gprw.data` and reset `gprw.subsample` and `gprw.GPR` -- very important!
  """
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

function subsample!(gprw::GPRWrap; indices::Union{Array{Int,1}, UnitRange{Int}})
  """
  Subsample `gprw.data` using `indices`
  """
  gprw.subsample = @view(gprw.data[indices,..])
  gprw.__subsample_set = true
  println(name("subsample!"), size(gprw.subsample,1), " subsampled")
  flush(stdout)
end

function subsample!(gprw::GPRWrap, thrsh::Int)
  """
  Randomly subsample `gprw.data` if `thrsh` is greater than number of points;
  otherwise, just use the whole `gprw.data`

  Convention: if `thrsh` < 0, then no subsampling either, i.e. use all data

  This function ignores `gprw.thrsh`
  """
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

function subsample!(gprw::GPRWrap)
  """
  Wrapper for subsample!(gprw::GPRWrap, thrsh:Int)
  """
  subsample!(gprw, gprw.thrsh)
end

function learn!(gprw::GPRWrap; kernel::String = "rbf", alpha = 0.5)
  if !gprw.__subsample_set
    println(warn("learn!"), "'subsample' is not set; attempting to set...")
    subsample!(gprw)
  end

  if kernel == "matern"
    GPR_kernel = 1.0 * Matern(length_scale = 1, nu = 1.5)
  else # including "rbf", which is the default
    if kernel != "rbf"
      println(warn("learn!"), "Kernel '", kernel, "' is not supported; ",
              "falling back to RBF")
    end
    GPR_kernel = 1.0 * RBF(1, (1e-10, 1e+6)) + WhiteKernel(1, (1e-10, 10))
  end

  gprw.GPR = GaussianProcessRegressor(
      kernel = GPR_kernel,
      n_restarts_optimizer = 7,
      alpha = alpha
      )
  sklearn.fit!(gprw.GPR, gprw.subsample[:,1:end-1], gprw.subsample[:,end])

  println(name("learn!"), gprw.GPR.kernel_)
  flush(stdout)
end

function predict(gprw::GPRWrap, x)
  """
  Add an extra dimension to `x` if it is a vector (scikit-learn's whim);
  return predicted values
  """
  if ndims(x) == 1
    return gprw.GPR.predict( reshape(x, (size(x)...,1)) )
  else
    return gprw.GPR.predict(x)
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


