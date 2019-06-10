using Parameters # lets you have defaults for fields

@with_kw mutable struct L96m
  """
  Lorenz '96 multiscale

  Parameters:
  - `K`   : number of slow variables
  - `J`   : number of fast variables per slow variable
  - `hx`  : coupling term (in equations for slow variables)
  - `hy`  : coupling term (in equations for fast variables)
  - `F`   : forcing term for slow variables
  - `eps` : scale separation term

  Other:
  - `G` : functional closure for slow variables (usually a GPR-closure)
  """
  K::Int = 9
  J::Int = 8
  hx::Float64 = -0.8
  hy::Float64 = 1
  F::Float64 = 10
  eps::Float64 = 2^(-7)
  G = nothing
end

function full(rhs::Array{<:Real,1}, z::Array{<:Real,1}, _s::L96m, t)
  """
  Compute full RHS of the Lorenz '96 multiscale system.
  The convention is that the first K variables are slow, while the rest K*J
  variables are fast.

  Input:
  - `z`   : vector of size (K + K*J)
  - `_s`  : parameters
  - `t`   : time (not used here since L96m is autonomous)

  Output:
  - `rhs` : RHS computed at `z`

  """

  K = _s.K
  J = _s.J
  x = @view(z[1:K])
  y = @view(z[K+1:end])

  ### slow variables subsystem ###
  # compute Yk averages
  Yk = compute_Yk(_s, z)

  # three boundary cases
  rhs[1] = -x[K]   * (x[K-1] - x[2]) - x[1]
  rhs[2] = -x[1]   * (x[K]   - x[3]) - x[2]
  rhs[K] = -x[K-1] * (x[K-2] - x[1]) - x[K]

  # general case
  rhs[3:K-1] = -x[2:K-2] .* (x[1:K-3] - x[4:K]) - x[3:K-1]

  # add forcing
  rhs[1:K] .+= _s.F

  # add coupling w/ fast variables via averages
  rhs[1:K] .+= _s.hx * Yk

  ### fast variables subsystem ###
  # three boundary cases
  rhs[K+1]   = -y[2]   * (y[3] - y[end]  ) - y[1]
  rhs[end-1] = -y[end] * (y[1] - y[end-2]) - y[end-1]
  rhs[end]   = -y[1]   * (y[2] - y[end-1]) - y[end]

  # general case
  rhs[K+2:end-2] = -y[3:end-1] .* (y[4:end] - y[1:end-3]) - y[2:end-2]

  # add coupling w/ slow variables
  for k in 1:K
    rhs[K+1 + (k-1)*J : K + k*J] .+= _s.hy * x[k]
  end

  # divide by epsilon
  rhs[K+1:end] ./= _s.eps

  return rhs
end

function compute_Yk(_s::L96m, z::Array{<:Real,1})
  """
  Reshape a vector of y_{j,k} into a matrix, then sum along one dim and divide
  by J to get averages
  """
  return dropdims(
      sum( reshape(z[_s.K+1:end], _s.J, _s.K), dims = 1 ),
      dims = 1
  ) / _s.J
end


