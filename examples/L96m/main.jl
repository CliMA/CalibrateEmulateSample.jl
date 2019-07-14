#!/usr/bin/julia --

import OrdinaryDiffEq
import PyPlot
import ConfParser
import ArgParse

const DE = OrdinaryDiffEq
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

"""
Prints information at the start of integration run
"""
function print_morning(name::String)
  print(rpad("(" * name * ")", RPAD))
end

"""
Prints information at the end of integration run
"""
function print_night(steps::Int, elapsed::Real)
  println("steps:", lpad(steps, LPAD_INTEGER),
          "\t\telapsed:", lpad(elapsed, LPAD_FLOAT))
end

function run_l96(rhs, ic, T; converging = false)
  pb = DE.ODEProblem(rhs, ic, (0.0, T), l96)
  return DE.solve(
                  pb,
                  SOLVER,
                  reltol = reltol,
                  abstol = abstol,
                  dtmax = dtmax,
                  save_everystep = !converging
                 )
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
# which runs to perform ########################################################
# run_conv     DNS, converging to attractor run, skipped; full system
# run_dns      DNS, direct numerical simulation; full system
# run_bal      balanced, naive closure; only slow variables
# run_reg      regressed, GPR closure; only slow variables
# run_onl      online, GPR closure learned iteratively; only slow variables
push!(parameters, "run_conv", "run_dns", "run_bal", "run_reg", "run_onl")

const run_conv = get_cfg_value(config, "runs", "run_conv", true)
const run_dns  = get_cfg_value(config, "runs", "run_dns",  true)
const run_bal  = get_cfg_value(config, "runs", "run_bal",  true)
const run_reg  = get_cfg_value(config, "runs", "run_reg",  true)
const run_onl  = get_cfg_value(config, "runs", "run_onl",  true)

# integration parameters #######################################################
# T            integration time
# T_conv       converging integration time
# T_lrnreg     time for gathering training data for regressed GP closure
# T_compile    force JIT compilation
push!(parameters, "T", "T_conv", "T_lrnreg", "T_compile")

const T = get_cfg_value(config, "integration", "T", 4.0)
const T_conv = get_cfg_value(config, "integration", "T_conv", 100.0)
const T_lrnreg = get_cfg_value(config, "integration", "T_lrnreg", 15.0)
const T_compile = 1e-10
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
set_G0(l96) # set the linear closure (as in balanced) -- needed for compilation

z00 = random_init(l96)
if run_reg || run_onl
  z00_lrn = random_init(l96)
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

# full L96m integration (learning data)
if run_reg || run_onl
  print_morning("full, learning, converging")
  elapsed_lrncv = @elapsed begin
    sol_lrncv = run_l96(full, z00_lrn, T_conv, converging = true)
  end
  print_night(length(sol_lrncv.t), elapsed_lrncv)
  z0_lrn = sol_lrncv[:,end]
end

# full L96m integration (converging to attractor)
if run_conv
  print_morning("full, converging")
  elapsed_conv = @elapsed begin
    sol_conv = run_l96(full, z00, T_conv, converging = true)
  end
  print_night(length(sol_conv.t), elapsed_conv)
  z0 = sol_conv[:,end]
end

# full L96m integration
if run_dns
  print_morning("full")
  elapsed_dns = @elapsed begin
    sol_dns = run_l96(full, z0, T)
  end
  print_night(length(sol_dns.t), elapsed_dns)
end

# balanced L96m integration
if run_bal
  print_morning("balanced")
  elapsed_bal = @elapsed begin
    sol_bal = run_l96(balanced, z0[1:l96.K], T)
  end
  print_night(length(sol_bal.t), elapsed_bal)
end

# regressed L96m integration
if run_reg
  print_morning("regressed")
  elapsed_reg = @elapsed begin
    sol_reg = run_l96(regressed, z0[1:l96.K], T)
  end
  print_night(length(sol_reg.t), elapsed_reg)
end

################################################################################
# plot section #################################################################
################################################################################
# plot DNS
if run_dns
  plt.plot(sol_dns.t, sol_dns[k,:], label = "DNS")
  plt.plot(sol_dns.t, sol_dns[l96.K + (k-1)*l96.J + j,:],
           lw = 0.6, alpha = 0.6, color="gray")
end

# plot balanced
if run_bal
  plt.plot(sol_bal.t, sol_bal[k,:], label = "balanced")
end

# plot regressed
if run_reg
  plt.plot(sol_reg.t, sol_reg[k,:], label = "regressed")
end

plt.legend()
plt.show()


