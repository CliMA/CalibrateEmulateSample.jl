#!/usr/bin/julia --

using DifferentialEquations
using PyPlot
include("L96m.jl")


################################################################################
# constants section ############################################################
################################################################################
EPS = 1e-14
DEBUG = false

T = 4 # integration time
T_conv = 10 # converging integration time
T_learn = 15 # time to gather training data for GP
T_hist = 10000 # time to gather histogram statistics

dt = 0.010 # maximum step size
tau = 1e-3 # maximum step size for histogram statistics
dt_conv = 0.01 # maximum step size for converging to attractor

k = 1 # index of the slow variable to save etc.
j = 2 # index of the fast variable to save/plot etc.

hx = -0.8

################################################################################
# IC section ###################################################################
################################################################################
l96 = L96m(hx = hx, J = 8)

#z0 = zeros(l96.K + l96.K * l96.J)
z0 = Array{Float64}(undef, l96.K + l96.K * l96.J)

# method 1
z0[1:l96.K] .= rand(l96.K) * 15 .- 5
for k_ in 1:l96.K
  z0[l96.K+1 + (k_-1)*l96.J : l96.K + k_*l96.J] .= z0[k_]
end

################################################################################
# main section #################################################################
################################################################################

# full L96m integration (converging to attractor)
prob = ODEProblem(full, z0, (0.0, T), l96)
sol = solve(prob, Tsit5(), reltol = 1e-3, abstol = 1e-6)

################################################################################
# plot section #################################################################
################################################################################
plot(sol.t, sol[k,:], label = "dns")
plot(sol.t, sol[l96.K + (k-1)*l96.J + j,:],
    lw = 0.6, alpha = 0.6, color="gray")
show()


