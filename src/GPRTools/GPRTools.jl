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
  data_sample = nothing
  GPR = nothing
  __set::Bool = false
  __sample_set::Bool = false
end

function set_data(gprw::GPRWrap, data::Array{<:Real})
  if ndims(data) > 2
    # TODO : how to slice a multidimensional array? [:,:, 1,..,1]
    #println("WARNING (set_data): ndims(data) > 2; will use the first two dims")
    throw(ErrorException("set_data: ndims(data) > 2; cannot proceed"))
  elseif ndims(data) < 2
    throw(ErrorException("set_data: ndims(data) < 2; cannot proceed"))
  end
  gprw.data = data
  gprw.__set = true
  println("set_data: success, number of data points: ", size(data,1))
  flush(stdout)
end

function set_sample(gprw::GPRWrap, thrsh::Int)
  if thrsh == 0
    throw(ErrorException("set_sample: 'thrsh' == 0, cannot sample"))
  end
  if !gprw.__set
    throw(ErrorException("set_sample: 'data' is not set, cannot sample"))
  end

  gprw.thrsh = thrsh
  N = size(gprw.data,1)
  if N > gprw.thrsh
    inds = StatsBase.sample(1:N, gprw.thrsh, replace = false)
  else
    inds = 1:N
  end
  gprw.data_sample = @view(gprw.data[inds,..])
  gprw.__sample_set = true
end

function set_sample(gprw::GPRWrap)
  """
  Wrapper for set_sample(gprw::GPRWrap, thrsh:Int)
  """
  set_sample(gprw, gprw.thrsh)
end

function learn(gprw::GPRWrap; kernel::String = "rbf")
  if !gprw.__sample_set
    throw(ErrorException("learn: 'sample_data' is not set"))
  end

  if kernel == "matern"
    GPR_kernel = 1.0 * Matern(length_scale = 3, nu = 1.5)
  else # including "rbf", which is the default
    if kernel != "rbf"
      println("WARNING (learn): Kernel '", kernel, "' is not supported; ",
              "falling back to RBF")
    end
    GPR_kernel = 1.0 * RBF(3, (1e-10, 1e+6)) + WhiteKernel()
  end

  gprw.GPR = GaussianProcessRegressor(
      kernel = GPR_kernel,
      n_restarts_optimizer = 7,
      alpha = 0.5
      )
  sklearn.fit!(gprw.GPR, gprw.data_sample[:,1:end-1], gprw.data_sample[:,end])

  println("learn: ", gprw.GPR.kernel_)
  flush(stdout)
end


