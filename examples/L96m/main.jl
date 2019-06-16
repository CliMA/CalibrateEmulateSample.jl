#!/usr/bin/julia --

import DifferentialEquations
import PyPlot
const DE = DifferentialEquations
const plt = PyPlot
include("L96m.jl")


################################################################################
# constants section ############################################################
################################################################################
const RPAD = 42
const LPAD_INTEGER = 7
const LPAD_FLOAT = 13
const SOLVER = DE.Tsit5()

const T = 4 # integration time
const T_compile = 1e-10 # force JIT compilation
const T_conv = 100 # converging integration time
#const T_learn = 15 # time to gather training data for GP
#const T_hist = 10000 # time to gather histogram statistics

# integrator parameters
const dtmax = 1e-3 # maximum step size
#const tau = 1e-3 # maximum step size for histogram statistics
#const dt_conv = 0.01 # maximum step size for converging to attractor
const reltol = 1e-3 # relative tolerance
const abstol = 1e-6 # absolute tolerance

const k = 1 # index of the slow variable to save etc.
const j = 1 # index of the fast variable to save/plot etc.

const hx = -0.8

################################################################################
# IC section ###################################################################
################################################################################
l96 = L96m(hx = hx, J = 8)

z00 = Array{Float64}(undef, l96.K + l96.K * l96.J)

# method 1
z00[1:l96.K] .= rand(l96.K) * 15 .- 5
for k_ in 1:l96.K
  z00[l96.K+1 + (k_-1)*l96.J : l96.K + k_*l96.J] .= z00[k_]
end

################################################################################
# main section #################################################################
################################################################################

# force compilation of functions used in numerical integration
print(rpad("(JIT compilation)", RPAD))
elapsed_jit = @elapsed begin
  pb_jit = DE.ODEProblem(full, z00, (0.0, T_compile), l96)
  DE.solve(pb_jit, SOLVER, reltol = reltol, abstol = abstol, dtmax = dtmax)
  pb_jit = DE.ODEProblem(balanced, z00[1:l96.K], (0.0, T_compile), l96)
  DE.solve(pb_jit, SOLVER, reltol = reltol, abstol = abstol, dtmax = dtmax)
end
println(" " ^ (LPAD_INTEGER + 6),
        "\t\telapsed:", lpad(elapsed_jit, LPAD_FLOAT))

# full L96m integration (converging to attractor)
print(rpad("(full, converging)", RPAD))
elapsed_conv = @elapsed begin
  pb_conv = DE.ODEProblem(full, z00, (0.0, T_conv), l96)
  sol_conv = DE.solve(pb_conv,
                      SOLVER,
                      reltol = reltol,
                      abstol = abstol,
                      dtmax = dtmax
                     )
end
println("steps:", lpad(length(sol_conv.t), LPAD_INTEGER),
        "\t\telapsed:", lpad(elapsed_conv, LPAD_FLOAT))
z0 = sol_conv[:,end]

# full L96m integration
print(rpad("(full)", RPAD))
elapsed_dns = @elapsed begin
  pb_dns = DE.ODEProblem(full, z0, (0.0, T), l96)
  sol_dns = DE.solve(pb_dns,
                     SOLVER,
                     reltol = reltol,
                     abstol = abstol,
                     dtmax = dtmax
                    )
end
println("steps:", lpad(length(sol_dns.t), LPAD_INTEGER),
        "\t\telapsed:", lpad(elapsed_dns, LPAD_FLOAT))

# balanced L96m integration
print(rpad("(balanced)", RPAD))
elapsed_bal = @elapsed begin
  pb_bal = DE.ODEProblem(balanced, z0[1:l96.K], (0.0, T), l96)
  sol_bal = DE.solve(pb_bal,
                     SOLVER,
                     reltol = reltol,
                     abstol = abstol,
                     dtmax = dtmax
                    )
end
println("steps:", lpad(length(sol_bal.t), LPAD_INTEGER),
        "\t\telapsed:", lpad(elapsed_bal, LPAD_FLOAT))

################################################################################
# plot section #################################################################
################################################################################
# plot DNS
plt.plot(sol_dns.t, sol_dns[k,:], label = "DNS")
plt.plot(sol_dns.t, sol_dns[l96.K + (k-1)*l96.J + j,:],
    lw = 0.6, alpha = 0.6, color="gray")

# plot balanced
plt.plot(sol_bal.t, sol_bal[k,:], label = "balanced")

plt.legend()
plt.show()


