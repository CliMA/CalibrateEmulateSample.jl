using Parameters # lets you have defaults for fields

@with_kw mutable struct L96m
  #predictor::Function
  K::Int = 9
  J::Int = 8
  hx::Float32 = -0.8
  hy::Float32 = 1
  F::Float32 = 10
  eps::Float32 = 2^(-7)
end

function full(rhs::Array{<:Real,1}, z::Array{<:Real,1}, _s::L96m, t)
  #= Full system RHS =#
  K = _s.K
  J = _s.J
  #rhs = Array{Float64}(K + K*J)
  x = z[1:K]
  y = z[K+1:end]

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
  rhs[1:K] += _s.hx * Yk

  ### fast variables subsystem ###
  # three boundary cases
  rhs[K+1]   = -y[2]   * (y[3] - y[end]  ) - y[1]
  rhs[end-1] = -y[end] * (y[1] - y[end-2]) - y[end-1]
  rhs[end]   = -y[1]   * (y[2] - y[end-1]) - y[end]

  # general case
  rhs[K+2:end-2] = -y[3:end-1] .* (y[4:end] - y[1:end-3]) - y[2:end-2]

  # add coupling w/ slow variables
  for k in 1:K
    rhs[K + (k-1)*J : K + k*J] .+= _s.hy * x[k]
  end

  # divide by epsilon
  rhs[K:end] ./= _s.eps

  return rhs
end

function compute_Yk(_s::L96m, z::Array{<:Real,1})
  # reshape a vector of y_{j,k} into a matrix, then sum along one dim and
  # divide by J to get averages
  return dropdims(
      sum( reshape(z[_s.K+1:end], _s.J, _s.K), dims = 1 ),
      dims = 1
  ) / _s.J
end


