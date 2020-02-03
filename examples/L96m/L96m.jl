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
@with_kw mutable struct L96m{FT<:AbstractFloat,I<:Int}
  K::I = 9
  J::I = 8
  hx::FT = -0.8
  hy::FT = 1.0
  F::FT = 10.0
  eps::FT = 2^(-7)
  G = nothing
end

"""
Compute full RHS of the Lorenz '96 multiscale system.
The convention is that the first K variables are slow, while the rest K*J
variables are fast.

Input:
 - `z`   : vector of size (K + K*J)
 - `p`   : parameters (L96m struct)
 - `t`   : time (not used here since L96m is autonomous)

Output:
 - `rhs` : RHS computed at `z`

"""
function full(rhs::Array{FT,1}, z::Array{FT,1}, p::L96m{FT,I}, t) where {FT,I}
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
 - `p`   : parameters (L96m struct)
 - `t`   : time (not used here since L96m is autonomous)

Output:
 - `rhs` : balanced RHS computed at `x`

"""
function balanced(rhs::Array{FT,1}, x::Array{FT,1}, p::L96m{FT,I}, t) where {FT,I}
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
 - `p`   : parameters (L96m struct)
 - `t`   : time (not used here since L96m is autonomous)

Output:
 - `rhs` : regressed RHS computed at `x`

"""
function regressed(rhs::Array{FT,1}, x::Array{FT,1}, p::L96m{FT,I}, t) where {FT,I}
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
function compute_Yk(p::L96m{FT,I}, z::Array{FT,1}) where {FT,I}
  return dropdims(
      sum( reshape(z[p.K+1:end], p.J, p.K), dims = 1 ),
      dims = 1
  ) / p.J
end

"""
Set the closure `p.G` to a linear one with slope `slope`.
If unspecified, slope is equal to `p.hy`.
"""
function set_G0(p::L96m{FT,I}; slope = nothing) where {FT,I}
  if (slope == nothing) || (!isa(slope, FT))
    slope = p.hy
  end
  p.G = x -> slope * x
end

"""
Wrapper for set_G0(p::L96m; slope = nothing).
"""
function set_G0(p::L96m{FT,I}, slope::FT) where {FT,I}
  set_G0(p, slope = slope)
end

"""
Gather (xk, Yk) pairs that are further used to train a GP regressor

Input:
 - `p`     : parameters (L96m struct)
 - `sol`   : time series of a solution; time steps are in the 2nd dimension
             (usually, just `sol` output from a time-stepper)

Output:
 - `pairs` : a 2-d array of size (N, 2) containing (xk, Yk) pairs, where N is
             the 2nd dimension in `sol` (number of time steps)

"""
function gather_pairs(p::L96m{FT,I}, sol) where {FT,I}
  N = size(sol, 2)
  pairs = Array{FT, 2}(undef, p.K * N, 2)
  for n in 1:N
    pairs[p.K * (n-1) + 1 : p.K * n, 1] = sol[1:p.K, n]
    pairs[p.K * (n-1) + 1 : p.K * n, 2] = compute_Yk(p, sol[:,n])
  end
  return pairs
end

"""
Returns a randomly initialized array that can be used as an IC to ODE solver

The returned array is of size `p.K + p.K * p.J`.
The first `p.K` variables are slow and are drawn randomly ~ U[-5; 10]; each of
the fast variables corresponding to the same slow variable is set to the value
of that slow variable.

For example, if K == 2 and J = 3, the returned array will be
  [ rand1, rand2, rand1, rand1, rand1, rand2, rand2, rand2 ]

Input:
 - `p`     : parameters (L96m struct)

Output:
 - `z00`   : array of size `p.K + p.K * p.J` with random values
"""
function random_init(p::L96m{FT,I}) where {FT,I}
  z00 = Array{FT}(undef, p.K + p.K * p.J)

  z00[1:p.K] .= rand(p.K) * 15 .- 5
  for k_ in 1:p.K
    z00[p.K+1 + (k_-1)*p.J : p.K + k_*p.J] .= z00[k_]
  end

  return z00
end


