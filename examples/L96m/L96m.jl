using Parameters # lets you have defaults for fields

"""
Lorenz '96 multiscale

Parameters:
 - `K`   : number of slow variables
 - `J`   : number of fast variables per slow variable
 - `hx`  : coupling term (in equations for slow variables)
 - `hy`  : coupling term (in equations for fast variables)
 - `F`   : forcing term for slow variables
 - `eps` : scale separation constant

Other:
 - `G` : functional closure for slow variables (usually a GPR-closure)
"""
@with_kw mutable struct L96m
  K::Int = 9
  J::Int = 8
  hx::Float64 = -0.8
  hy::Float64 = 1.0
  F::Float64 = 10.0
  eps::Float64 = 2^(-7)
  G = nothing
end

"""
Compute full RHS of the Lorenz '96 multiscale system.
The convention is that the first K variables are slow, while the rest K*J
variables are fast.

Input:
 - `z`   : vector of size (K + K*J)
 - `p`   : parameters
 - `t`   : time (not used here since L96m is autonomous)

Output:
 - `rhs` : RHS computed at `z`

"""
function full(rhs::Array{<:Real,1}, z::Array{<:Real,1}, p::L96m, t)
  K = p.K
  J = p.J
  x = @view(z[1:K])
  y = @view(z[K+1:end])

  ### slow variables subsystem ###
  # compute Yk averages
  Yk = compute_Yk(p, z)

  # three boundary cases
  rhs[1] = -x[K]   * (x[K-1] - x[2]) - x[1]
  rhs[2] = -x[1]   * (x[K]   - x[3]) - x[2]
  rhs[K] = -x[K-1] * (x[K-2] - x[1]) - x[K]

  # general case
  rhs[3:K-1] = -x[2:K-2] .* (x[1:K-3] - x[4:K]) - x[3:K-1]

  # add forcing
  rhs[1:K] .+= p.F

  # add coupling w/ fast variables via averages
  rhs[1:K] .+= p.hx * Yk

  ### fast variables subsystem ###
  # three boundary cases
  rhs[K+1]   = -y[2]   * (y[3] - y[end]  ) - y[1]
  rhs[end-1] = -y[end] * (y[1] - y[end-2]) - y[end-1]
  rhs[end]   = -y[1]   * (y[2] - y[end-1]) - y[end]

  # general case
  rhs[K+2:end-2] = -y[3:end-1] .* (y[4:end] - y[1:end-3]) - y[2:end-2]

  # add coupling w/ slow variables
  for k in 1:K
    rhs[K+1 + (k-1)*J : K + k*J] .+= p.hy * x[k]
  end

  # divide by epsilon
  rhs[K+1:end] ./= p.eps

  return rhs
end

"""
Compute balanced RHS of the Lorenz '96 multiscale system; i.e. only slow
variables with the linear closure.
Both `rhs` and `x` are vectors of size p.K.

Input:
 - `x`   : vector of size K
 - `p`   : parameters
 - `t`   : time (not used here since L96m is autonomous)

Output:
 - `rhs` : balanced RHS computed at `x`

"""
function balanced(rhs::Array{<:Real,1}, x::Array{<:Real,1}, p::L96m, t)
  K = p.K

  # three boundary cases
  rhs[1] = -x[K]   * (x[K-1] - x[2]) - (1 - p.hx*p.hy) * x[1]
  rhs[2] = -x[1]   * (x[K]   - x[3]) - (1 - p.hx*p.hy) * x[2]
  rhs[K] = -x[K-1] * (x[K-2] - x[1]) - (1 - p.hx*p.hy) * x[K]

  # general case
  rhs[3:K-1] = -x[2:K-2] .* (x[1:K-3] - x[4:K]) - (1 - p.hx*p.hy) * x[3:K-1]

  # add forcing
  rhs .+= p.F

  return rhs
end

"""
Compute slow-variable closed RHS of the Lorenz '96 Multiscale system;
i.e. only slow variables with some closure instead of Yk.
Closure is taken from p.G.
Both `rhs` and `x` are vectors of size p.K.

Input:
 - `x`   : vector of size K
 - `p`   : parameters
 - `t`   : time (not used here since L96m is autonomous)

Output:
 - `rhs` : regressed RHS computed at `x`

"""
function regressed(rhs::Array{<:Real,1}, x::Array{<:Real,1}, p::L96m, t)
  K = p.K

  # three boundary cases
  rhs[1] = -x[K]   * (x[K-1] - x[2]) - x[1]
  rhs[2] = -x[1]   * (x[K]   - x[3]) - x[2]
  rhs[K] = -x[K-1] * (x[K-2] - x[1]) - x[K]

  # general case
  rhs[3:K-1] = -x[2:K-2] .* (x[1:K-3] - x[4:K]) - x[3:K-1]

  # add forcing
  rhs .+= p.F

  # add closure
  rhs .+= p.hx * p.G(x)

  return rhs
end

"""
Reshape a vector of y_{j,k} into a matrix, then sum along one dim and divide
by J to get averages
"""
function compute_Yk(p::L96m, z::Array{<:Real,1})
  return dropdims(
      sum( reshape(z[p.K+1:end], p.J, p.K), dims = 1 ),
      dims = 1
  ) / p.J
end

"""
Set the closure `p.G` to a linear one with slope `slope`.
If unspecified, slope is equal to `p.hy`.
"""
function set_G0(p::L96m; slope = nothing)
  if (slope == nothing) || (!isa(slope, Real))
    slope = p.hy
  end
  p.G = x -> slope * x
end

"""
Wrapper for set_G0(p::L96m; slope = nothing).
"""
function set_G0(p::L96m, slope::Real)
  set_G0(p, slope = slope)
end


