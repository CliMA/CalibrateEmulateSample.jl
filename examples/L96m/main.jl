#!/usr/bin/julia --

import DifferentialEquations
import PyPlot
import ConfParser
import ArgParse

const DE = DifferentialEquations
const plt = PyPlot
const CP = ConfParser
const AP = ArgParse
include("L96m.jl")

################################################################################
# function section #############################################################
################################################################################
function parse_cli_args()
  ap_settings = AP.ArgParseSettings()
  AP.@add_arg_table ap_settings begin
    "--cfg", "-c"
      help = "name of the ini-file to be used as config"
      arg_type = String
      default = "cfg.ini"
  end

  return AP.parse_args(ap_settings)
end

function get_cfg_value(config::CP.ConfParse, sect::String, key::String, default)
  return try
    val = CP.retrieve(config, sect, key)
    if isa(val, String) && String != typeof(default)
      parse(typeof(default), val)
    else
      val
    end
  catch
    default
  end
end

function print_var(var_name::String; offset::Int = 0)
  println(
          " "^offset,
          rpad(var_name, RPAD_VAR),
          lpad(eval(Symbol(var_name)), LPAD_VAR)
         )
end

function print_var(var_names::Array{String})
  println("# Parameters")
  for var_name in var_names
    print_var(var_name, offset = 2)
  end
end

################################################################################
# config section ###############################################################
################################################################################
args_dict = parse_cli_args()
config = try
  CP.ConfParse(args_dict["cfg"])
catch
  CP.ConfParse("cfg.ini")
end

CP.parse_conf!(config)

################################################################################
# parameters section ###########################################################
################################################################################
const RPAD = 42
const RPAD_VAR = 21
const LPAD_INTEGER = 7
const LPAD_FLOAT = 13
const LPAD_VAR = 29

parameters = String[]
# integration parameters #######################################################
# T            integration time
# T_conv       converging integration time
# T_compile    force JIT compilation
push!(parameters, "T", "T_conv", "T_compile")

const T = get_cfg_value(config, "integration", "T", 4.0)
const T_conv = get_cfg_value(config, "integration", "T_conv", 100.0)
const T_compile = 1e-10
#const T_learn = 15 # time to gather training data for GP
#const T_hist = 10000 # time to gather histogram statistics

# time-stepper parameters ######################################################
# SOLVER       time-stepper itself
# dtmax        maximum step size
# reltol       relative tolerance
# abstol       absolute tolerance
push!(parameters, "SOLVER", "dtmax", "reltol", "abstol")

const SOLVER_STR = get_cfg_value(config, "integration", "SOLVER", "Tsit5")
const SOLVER = getfield(DE, Symbol(SOLVER_STR))()
const dtmax = get_cfg_value(config, "integration", "dtmax", 1e-3)
#const tau = 1e-3 # maximum step size for histogram statistics
#const dt_conv = 0.01 # maximum step size for converging to attractor
const reltol = get_cfg_value(config, "integration", "reltol", 1e-3)
const abstol = get_cfg_value(config, "integration", "abstol", 1e-6)

# save/plot parameters #########################################################
const k = 1 # index of the slow variable to save etc.
const j = 1 # index of the fast variable to save/plot etc.
push!(parameters, "k", "j")

# L96 parameters ###############################################################
const hx = -0.8
push!(parameters, "hx")

print_var(parameters)

################################################################################
# IC section ###################################################################
################################################################################
l96 = L96m(hx = hx, J = 8)
set_G0(l96) # set the linear closure (essentially, as in balanced)

z00 = Array{Float64}(undef, l96.K + l96.K * l96.J)

# method 1
z00[1:l96.K] .= rand(l96.K) * 15 .- 5
for k_ in 1:l96.K
  z00[l96.K+1 + (k_-1)*l96.J : l96.K + k_*l96.J] .= z00[k_]
end

################################################################################
# main section #################################################################
################################################################################
println("# Main")

# force compilation of functions used in numerical integration
print(rpad("(JIT compilation)", RPAD))
elapsed_jit = @elapsed begin
  pb_jit = DE.ODEProblem(full, z00, (0.0, T_compile), l96)
  DE.solve(pb_jit, SOLVER, reltol = reltol, abstol = abstol, dtmax = dtmax)
  pb_jit = DE.ODEProblem(balanced, z00[1:l96.K], (0.0, T_compile), l96)
  DE.solve(pb_jit, SOLVER, reltol = reltol, abstol = abstol, dtmax = dtmax)
  pb_jit = DE.ODEProblem(regressed, z00[1:l96.K], (0.0, T_compile), l96)
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

# regressed L96m integration
print(rpad("(regressed)", RPAD))
elapsed_reg = @elapsed begin
  pb_reg = DE.ODEProblem(regressed, z0[1:l96.K], (0.0, T), l96)
  sol_reg = DE.solve(pb_reg,
                     SOLVER,
                     reltol = reltol,
                     abstol = abstol,
                     dtmax = dtmax
                    )
end
println("steps:", lpad(length(sol_reg.t), LPAD_INTEGER),
        "\t\telapsed:", lpad(elapsed_reg, LPAD_FLOAT))

################################################################################
# plot section #################################################################
################################################################################
# plot DNS
plt.plot(sol_dns.t, sol_dns[k,:], label = "DNS")
plt.plot(sol_dns.t, sol_dns[l96.K + (k-1)*l96.J + j,:],
    lw = 0.6, alpha = 0.6, color="gray")

# plot balanced
plt.plot(sol_bal.t, sol_bal[k,:], label = "balanced")

# plot regressed
plt.plot(sol_reg.t, sol_reg[k,:], label = "regressed")

plt.legend()
plt.show()


