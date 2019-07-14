module GPR
"""
For the time being, please use `include("src/GPR.jl")` and not `using Solus.GPR`
since there are precompile issues with the backend (scikit-learn)
"""

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
 - mmstd (1 method)
 - plot_fit (1 method)

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

"""
Return mesh, mean and st. deviation over the whole data range

Computes min and max of `gprw.data` x-range and returns equispaced mesh with
`mesh_n` number of points, mean and st. deviation computed over that mesh

Parameters:
  - gprw:        an instance of GPR.Wrap
  - mesh_n:      number of mesh points (1001 by default)

Returns:
  - (m, m, std): mesh, mean and st. deviation
"""
function mmstd(gprw::Wrap; mesh_n = 1001)
  mesh = range(minimum(gprw.data, dims=1)[1],
               maximum(gprw.data, dims=1)[1],
               length = mesh_n)
  return (mesh, predict(gprw, mesh, return_std = true)...)
end

"""
Plot mean (and 95% interval) along with data and subsample

The flag `plot_95` controls whether to plot 95% interval; `label` may provide
labels in the following order:
  data points, subsample, GP mean, 95% interval (if requested)

If you're using Plots.jl and running it from a file rather than REPL, you need
to wrap the call:
`display(GPR.plot_fit(gprw, Plots))`

Parameters:
  - gprw:        an instance of GPR.Wrap
  - plt:         a module used for plotting (only PyPlot & Plots supported)
  - plot_95:     boolean flag, whether to plot 95% confidence interval
  - label:       a 3- or 4-tuple or vector of strings (no label by default)
"""
function plot_fit(gprw::Wrap, plt; plot_95 = false, label = nothing)
  if !gprw.__data_set
    println(warn("plot_fit"), "data is not set, nothing to plot")
    return
  end
  is_pyplot = (Symbol(plt) == :PyPlot)
  is_plots = (Symbol(plt) == :Plots)
  if !is_pyplot && !is_plots
    println(warn("plot_fit"), "only PyPlot & Plots are supported; not plotting")
    return
  end

  # set `cols` Dict with colors of the plots
  alpha_95 = 0.3 # alpha channel for shaded region, i.e. 95% interval
  cols = Dict{String, Any}()
  cols["mean"] = "black"
  if is_pyplot
    cols["data"] = "tab:gray"
    cols["sub"] = "tab:red"
    cols["shade"] = (0, 0, 0, alpha_95)
  elseif is_plots
    cols["data"] = "#7f7f7f" # tab10 gray
    cols["sub"] = "#d62728" # tab10 red
  end

  # set keyword argument dictionaries for plotting functions
  kwargs_data = Dict{Symbol, Any}()
  kwargs_sub  = Dict{Symbol, Any}()
  kwargs_mean = Dict{Symbol, Any}()
  kwargs_95   = Dict{Symbol, Any}()
  kwargs_aux  = Dict{Symbol, Any}()

  kwargs_data[:color]   = cols["data"]
  kwargs_sub[:color]    = cols["sub"]
  kwargs_mean[:color]   = cols["mean"]
  kwargs_mean[:lw]      = 2.5
  if is_pyplot
    kwargs_data[:ms]      = 4
    kwargs_sub[:ms]       = 4
    kwargs_95[:facecolor] = cols["shade"]
    kwargs_95[:edgecolor] = cols["mean"]
    kwargs_95[:lw]        = 0.5
    kwargs_95[:zorder]    = 10
  elseif is_plots
    kwargs_data[:ms]      = 2
    kwargs_sub[:ms]       = 2
    kwargs_95[:color]     = cols["mean"]
    kwargs_95[:fillalpha] = alpha_95
    kwargs_95[:lw]        = 2.5
    kwargs_95[:z]         = 10
    kwargs_aux[:color]    = cols["mean"]
    kwargs_aux[:lw]       = 0.5
    kwargs_aux[:label]    = ""
  end

  if label != nothing
    kwargs_data[:label] = label[1]
    kwargs_sub[:label]  = label[2]
    kwargs_mean[:label] = label[3]
    if is_pyplot
      kwargs_95[:label] = label[4]
    elseif is_plots
      kwargs_95[:label] = label[3]
    end
  elseif is_plots
    kwargs_data[:label] = ""
    kwargs_sub[:label]  = ""
    kwargs_mean[:label] = ""
    kwargs_95[:label]   = ""
  end


  mesh, mean, std = mmstd(gprw)

  # plot data, subsample and mean
  if is_pyplot
    plt.plot(gprw.data[:,1],      gprw.data[:,2], ".";      kwargs_data...)
    plt.plot(gprw.subsample[:,1], gprw.subsample[:,2], "."; kwargs_sub...)
    if plot_95
      plt.fill_between(mesh,
                       mean - 1.96 * std,
                       mean + 1.96 * std;
                       kwargs_95...)
    end
    plt.plot(mesh, mean; kwargs_mean...)
  elseif is_plots
    plt.scatter!(gprw.data[:,1],      gprw.data[:,2];      kwargs_data...)
    plt.scatter!(gprw.subsample[:,1], gprw.subsample[:,2]; kwargs_sub...)
    if plot_95
      plt.plot!(mesh,
                mean,
                ribbon = (1.96 * std, 1.96 * std);
                kwargs_95...)
      plt.plot!(mesh, mean - 1.96 * std; kwargs_aux...)
      plt.plot!(mesh, mean + 1.96 * std; kwargs_aux...)
    else
      plt.plot!(mesh, mean; kwargs_mean...)
    end
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


