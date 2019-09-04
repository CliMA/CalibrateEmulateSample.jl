using Parameters # lets you have defaults for fields

"""
Lorenz '96 multiscale

Parameters:
 - K:    number of slow variables
 - J:    number of fast variables per slow variable
 - hx:   coupling term (in equations for slow variables)
 - hy:   coupling term (in equations for fast variables)
 - F:    forcing term for slow variables
 - eps:  scale separation constant
 - k0:   L96m region to use in filtered integration (slow index)

Other:
 - G:    functional closure for slow variables (usually a GPR-closure)
"""
# TODO : make immutable; implement immutable L96mClosed that inherits L96m;
# maybe use constructors (?)
@with_kw mutable struct L96m
  K::Int
  J::Int
  hx::Array{Float64}
  hy::Float64
  F::Float64
  eps::Float64
  k0::UInt
  G::Any
end

const L96M_DNS = UInt8(0b000001)
const L96M_BAL = UInt8(0b000010)
const L96M_REG = UInt8(0b000100)
const L96M_ONL = UInt8(0b001000)
const L96M_FLT = UInt8(0b010000)
const L96M_FST = UInt8(0b100000)

"""
Return a L96m id based on the name

Input:
 - name:    string

Output:
 - id:      one of the UInt8 constants (L96M_DNS etc.)

"""
function l96m_id(name::String)
  id = UInt8(0)
  # r"string"i is the same as Regex("string", "i"), which means case-insensitive
  if occursin(r"dns"i, name) || occursin(r"full"i, name)
    id = L96M_DNS
  elseif occursin(r"bal"i, name)
    id = L96M_BAL
  elseif occursin(r"reg"i, name)
    id = L96M_REG
  elseif occursin(r"onl"i, name)
    id = L96M_ONL
  elseif occursin(r"flt"i, name) || occursin(r"filtered"i, name)
    id = L96M_FLT
  elseif occursin(r"fst"i, name) || occursin(r"fast"i, name)
    id = L96M_FST
  end
  return id
end

"""
Return a L96m plotting color based on the id

Input:
 - id:      one of the UInt8 constants (L96M_DNS etc.)

Output:
 - color:   string, either name of a color or its hex value

"""
function l96m_color(id::UInt8)
  color = "black"
  if id == L96M_DNS
    color = "#0072b2"
  elseif id == L96M_BAL
    color = "#d55e00"
  elseif id == L96M_REG
    color = "#009e73"
  elseif id == L96M_ONL
    color = "#cc79a7"
  elseif id == L96M_FLT
    color = "tab:pink"
  elseif id == L96M_FST
    color = "tab:gray"
  else
    println("WARNING (l96m_color): id not recognized")
  end
  return color
end

"""
Plot timeseries of a solution

Input:
 - p:       parameters (L96m struct)
 - plt:     a module used for plotting (currently has to be PyPlot)
 - sol:     solution timeseries (ODESolution) with time steps in row direction
 - k:       integer in [1, p.K]; slow variable to plot
 - label:   string
 - j:       integer in [1, p.J]; fast variable to plot (if requested)
 - fast:    boolean flag; whether to plot fast variable (only in DNS case)

"""
function plot_solution(p::L96m, plt, sol;
                       k, label::String, j = 1, fast = false, acf = false)
  id = l96m_id(label)
  color = l96m_color(id)
  if !acf
    plt.plot(sol.t, sol[k, : ], label = label, color = color)
  else
    plt.plot(sol[k, : ], label = label, color = color)
  end
  if id == L96M_DNS && fast
    if !acf
      plt.plot(sol.t, sol[p.K + (k-1)*p.J + j, : ],
               lw = 0.6, alpha = 0.6, color = l96m_color(L96M_FST))
    else
      plt.plot(sol[p.K + (k-1)*p.J + j, : ],
               lw = 0.6, alpha = 0.6, color = l96m_color(L96M_FST))
    end
  end
end

"""
Compute full RHS of the L96m system.
The convention is that the first K variables are slow, while the rest K*J
variables are fast.

Input:
 - z:    vector of size (K + K*J)
 - p:    parameters (L96m struct)
 - t:    time (not used here since L96m is autonomous)

Output:
 - rhs:  RHS computed at `z`

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
  rhs[1:K] .+= p.hx .* Yk

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
Compute balanced RHS of the L96m system; i.e. only slow
variables with the linear closure.
Both `rhs` and `x` are vectors of size `p.K`.

Input:
 - x:    vector of size K
 - p:    parameters (L96m struct)
 - t:    time (not used here since L96m is autonomous)

Output:
 - rhs:  balanced RHS computed at `x`

"""
function balanced(rhs::Array{<:Real,1}, x::Array{<:Real,1}, p::L96m, t)
  K = p.K

  # three boundary cases
  rhs[1] = -x[K]   * (x[K-1] - x[2]) - (1 - p.hx[1]*p.hy) * x[1]
  rhs[2] = -x[1]   * (x[K]   - x[3]) - (1 - p.hx[2]*p.hy) * x[2]
  rhs[K] = -x[K-1] * (x[K-2] - x[1]) - (1 - p.hx[K]*p.hy) * x[K]

  # general case
  rhs[3:K-1] = -x[2:K-2] .* (x[1:K-3] - x[4:K])
               - (1 .- p.hx[3:K-1] * p.hy) .* x[3:K-1]

  # add forcing
  rhs .+= p.F

  return rhs
end

"""
Compute slow-variable closed RHS of the L96m system;
i.e. only slow variables with some closure instead of Yk.
Closure is taken from `p.G`.
Both `rhs` and `x` are vectors of size `p.K`.

Input:
 - x:    vector of size K
 - p:    parameters (L96m struct)
 - t:    time (not used here since L96m is autonomous)

Output:
 - rhs:  regressed RHS computed at `x`

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
  rhs .+= p.hx .* p.G(x)

  return rhs
end

"""
Compute slow-variable RHS of the L96m system with a closure and one L96m region

Closure is taken from `p.G`, the region number is provided by `p.k0`.
RHS is computed using the closure everywhere except `p.k0`, where fast variables
are looped on themselves at the boundaries.
Both `rhs` and `z` are vectors of size `p.K + p.J`.

Input:
 - z:    vector of size K + J
 - p:    parameters (L96m struct)
 - t:    time (not used here since L96m is autonomous)

Output:
 - rhs:  filtered RHS computed at `z`

"""
function filtered(rhs::Array{<:Real,1}, z::Array{<:Real,1}, p::L96m, t)
  K = p.K
  J = p.J
  x = @view(z[1:K])
  y = @view(z[K+1:end])

  ### slow variables subsystem ###
  # compute Yk0 average
  Yk0 = compute_Yk(p, z)

  # three boundary cases
  rhs[1] = -x[K]   * (x[K-1] - x[2]) - x[1]
  rhs[2] = -x[1]   * (x[K]   - x[3]) - x[2]
  rhs[K] = -x[K-1] * (x[K-2] - x[1]) - x[K]

  # general case
  rhs[3:K-1] = -x[2:K-2] .* (x[1:K-3] - x[4:K]) - x[3:K-1]

  # add forcing
  rhs[1:K] .+= p.F

  # add coupling w/ fast variables via average for k0
  rhs[p.k0] += p.hx[p.k0] * Yk0[1] # Yk0 is a 1-element 1-dimensional array
  # add coupling w/ the rest via closure
  idx_wo_k0 = [ 1:(p.k0-1); (p.k0+1):K ]
  rhs[idx_wo_k0] .+= p.hx[idx_wo_k0] .* p.G(x[idx_wo_k0])

  ### fast variables subsystem ###
  # three boundary cases
  rhs[K+1]   = -y[2]   * (y[3] - y[end]  ) - y[1]
  rhs[end-1] = -y[end] * (y[1] - y[end-2]) - y[end-1]
  rhs[end]   = -y[1]   * (y[2] - y[end-1]) - y[end]

  # general case
  rhs[K+2:end-2] = -y[3:end-1] .* (y[4:end] - y[1:end-3]) - y[2:end-2]

  # add coupling w/ the k0 slow variable
  rhs[K+1:end] .+= p.hy * x[p.k0]

  # divide by epsilon
  rhs[K+1:end] ./= p.eps
end

"""
Compute Yk averages

Reshape a vector of y_{j,k} into a matrix, then sum along one dim and divide
by J to get averages
"""
function compute_Yk(p::L96m, z::Array{<:Real,1})
  r = div(length(z) - p.K, p.J) # number of regions with fast variables
  return dropdims(
      sum( reshape(z[p.K+1:end], p.J, r), dims = 1 ),
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

"""
Gather (xk, Yk) pairs that are further used to train a GPR

Input:
 - p:      parameters (L96m struct)
 - sol:    solution timeseries (ODESolution) with time steps in row direction

Output:
 - pairs:  2-d array of size (p.K * N, 2) containing (xk, Yk) pairs, where N is
             the 2nd dimension in `sol` (number of time steps)

"""
function gather_pairs(p::L96m, sol)
  N = size(sol, 2)
  pairs = Array{Float64, 2}(undef, p.K * N, 2)
  for n in 1:N
    pairs[p.K * (n-1) + 1 : p.K * n, 1] = sol[1:p.K, n]
    pairs[p.K * (n-1) + 1 : p.K * n, 2] = compute_Yk(p, sol[:,n])
  end
  return pairs
end

"""
Gather (xk0, Yk0) pairs at k0 region that are further used to train a GPR

The k0 region is provided by `p.k0`.

Input:
 - p:      parameters (L96m struct)
 - sol:    solution timeseries (ODESolution) with time steps in row direction

Output:
 - pairs:  2-d array of size (N, 2) containing (xk0, Yk0) pairs, where N is
             the 2nd dimension in `sol` (number of time steps)

"""
function gather_pairs_k0(p::L96m, sol)
  N = size(sol, 2)
  pairs = Array{Float64, 2}(undef, N, 2)
  pairs[1:end, 1] = sol[p.k0, 1:end]
  for n in 1:N
    # TODO : is it safe to use compute_Yk?
    #pairs[n, 2] = 1.0 * dropdims(compute_Yk(p, sol[:,n]), dims = 1)
    pairs[n, 2] = sum(sol[p.K+1:end, n]) / p.J
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
 - p:      parameters (L96m struct)

Output:
 - z00:    array of size `p.K + p.K * p.J` with random values

"""
function random_init(p::L96m)
  z00 = Array{Float64}(undef, p.K + p.K * p.J)

  z00[1:p.K] .= rand(p.K) * 15 .- 5
  for k_ in 1:p.K
    z00[p.K+1 + (k_-1)*p.J : p.K + k_*p.J] .= z00[k_]
  end

  return z00
end

function random_init2(p::L96m)
  z00 = Array{Float64}(undef, p.K + p.K * p.J)

  z00[1:p.K] .= rand(p.K) * 15 .- 5
  reshape(@view(z00[p.K+1 : end]), p.J, p.K)' .= rand(p.K)*15 .- 5

  return z00
end


