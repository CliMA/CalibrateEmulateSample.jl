#!/usr/bin/julia --

import OrdinaryDiffEq
import PyPlot
import ConfParser
import ArgParse
import NPZ

const DE = OrdinaryDiffEq
const plt = PyPlot
const CP = ConfParser
const AP = ArgParse
include("L96m.jl")
include("../../src/GPR.jl")

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
  println("\n# Parameters")
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
# All of the following are boolean flags:
# run_conv       DNS, converging to attractor run, skipped; full system
# run_dns        DNS, direct numerical simulation; full system
# run_bal        balanced, naive closure; only slow variables
# run_reg        regressed, GPR closure; only slow variables
# run_onl        online, GPR closure learned iteratively; only slow variables
push!(parameters, "run_conv", "run_dns", "run_bal", "run_reg", "run_onl")

const run_conv = get_cfg_value(config, "runs", "run_conv", true)
const run_dns  = get_cfg_value(config, "runs", "run_dns",  true)
const run_bal  = get_cfg_value(config, "runs", "run_bal",  true)
const run_reg  = get_cfg_value(config, "runs", "run_reg",  true)
const run_onl  = get_cfg_value(config, "runs", "run_onl",  true)

# integration parameters #######################################################
# T              integration time
# T_conv         converging integration time
# T_lrnreg       time for gathering training data for regressed GP closure
# T_compile      force JIT compilation
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

# online parameters (if online is requested) ###################################
# N_flt          number of filtering iterations (learning online closure)
# T_flt[n]       integration time of each filtering chunk, n = 1,..,N_flt
if run_onl
  push!(parameters, "N_flt", "T_flt")
  const N_flt = get_cfg_value(config, "online", "N_flt", 3)
  const T_flt = Array{Float64}(undef, 0)
  for n in 1:N_flt
    push!(T_flt, get_cfg_value(config, "online", "T_" * string(n), 10.0))
  end
end

# save/plot parameters #########################################################
# plot_GPR_reg     boolean; run `GPR.plot_fit` for regressed
# plot_GPR_flt     boolean; run `GPR.plot_fit` for filtered on each iteration
push!(parameters, "plot_GPR_reg", "plot_GPR_flt", "k", "j")
const plot_GPR_reg = get_cfg_value(config, "plotting", "plot_GPR_reg", false)
const plot_GPR_flt = get_cfg_value(config, "plotting", "plot_GPR_flt", false)
const k = 1 # index of the slow variable to save etc.
const j = 1 # index of the fast variable to save/plot etc.

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
  z00_lrn = random_init2(l96)
end
if run_reg
  gp_reg = GPR.Wrap(thrsh = 500)
end
if run_onl
  gp_flt = GPR.Wrap(thrsh = 500)
end

################################################################################
# main section #################################################################
################################################################################
println("\n# Main")

# force compilation of functions used in numerical integration
print(rpad("(JIT compilation)", RPAD))
elapsed_jit = @elapsed begin
  run_l96(full, z00, T_compile; converging = true)
  run_l96(balanced, z00[1:l96.K], T_compile; converging = true)
  run_l96(regressed, z00[1:l96.K], T_compile; converging = true)
  run_l96(filtered, z00[1:l96.K+l96.J], T_compile; converging = true)
end
println(" " ^ (LPAD_INTEGER + 6),
        "\t\telapsed:", lpad(elapsed_jit, LPAD_FLOAT))

# full L96m integration (converging for learning runs)
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
  print_morning("full, learning for regressed")
  elapsed_lrn = @elapsed begin
    sol_lrn = run_l96(full, z0_lrn, T_lrnreg)
  end
  print_night(length(sol_lrn.t), elapsed_lrn)
  GPR.set_data!(gp_reg, gather_pairs(l96, sol_lrn))
  GPR.subsample!(gp_reg)
  GPR.learn!(gp_reg)
  l96.G = x -> GPR.predict(gp_reg, x)
  if plot_GPR_reg
    GPR.plot_fit(gp_reg, plt, plot_95 = false,
                 label = ("Data", "Training", "Mean", "95% interval"))
    plt.show()
  end

  print_morning("regressed")
  elapsed_reg = @elapsed begin
    sol_reg = run_l96(regressed, z0[1:l96.K], T)
  end
  print_night(length(sol_reg.t), elapsed_reg)
end

# filtered + online L96m integration
if run_onl
  idx_flt = [ 1:l96.K; (l96.K + 1 + (l96.k0 - 1)*l96.J):(l96.K + l96.k0*l96.J) ]
  z0_flt = z0_lrn[idx_flt]
  #=
    NPZ.npzwrite("z0_flt.npy", z0_flt)
    z0_flt = NPZ.npzread("z0_flt.npy")
  =#
  #set_G0(l96, slope = -1)
  set_G0(l96)

  #=
  print_morning("filtered, learning for online")
  elapsed_flt = @elapsed begin
    sol_flt = run_l96(filtered, z0_flt, T_flt[1])
  end
  print_night(length(sol_flt.t), elapsed_flt)
    NPZ.npzwrite("sol_flt.npy", sol_flt[1:end, 1:end])
    NPZ.npzwrite("sol_flt_t.npy", sol_flt.t)
  GPR.set_data!(gp_flt, gather_pairs_k0(l96, sol_flt))
    #NPZ.npzwrite("pairs.npy", gp_flt.data)
  GPR.subsample!(gp_flt)
  GPR.learn!(gp_flt)
  l96.G = x -> GPR.predict(gp_flt, x)
  if plot_GPR_flt
    plt.figure(figsize = (5, 5), dpi = 160)
    GPR.plot_fit(gp_flt, plt, plot_95 = false,
                 label = ("Data", "Training", "Mean", "95% interval"))
    plt.legend()
    plt.show()
  end
  =#
  for n in 1:N_flt
    print_morning("filtered, learning for online, " * string(n))
    elapsed_flt = @elapsed begin
      sol_flt = run_l96(filtered, z0_flt, T_flt[n]; stiff = true)
    end
    print_night(length(sol_flt.t), elapsed_flt)
      NPZ.npzwrite("sol_flt_" * string(n) * ".npy", sol_flt[1:end, 1:end])
      NPZ.npzwrite("sol_flt_t_" * string(n) * ".npy", sol_flt.t)
    #global z0_flt = sol_flt[:,end]
    GPR.set_data!(gp_flt, gather_pairs_k0(l96, sol_flt))
    GPR.subsample!(gp_flt)
    GPR.learn!(gp_flt)
    l96.G = x -> GPR.predict(gp_flt, x)
    if plot_GPR_flt
      plt.figure(figsize = (5, 5), dpi = 160)
      GPR.plot_fit(gp_flt, plt, plot_95 = false,
                   label = ("Data", "Training", "Mean", "95% interval"))
      plt.legend()
      plt.show()
    end
  end

  print_morning("online")
  elapsed_onl = @elapsed begin
    sol_onl = run_l96(regressed, z0[1:l96.K], T)
  end
  print_night(length(sol_onl.t), elapsed_onl)
end

################################################################################
# plot section #################################################################
################################################################################
# plot DNS
if run_dns
  plot_solution(l96, plt, sol_dns, k = k, label = "DNS")
end

# plot balanced
if run_bal
  plot_solution(l96, plt, sol_bal, k = k, label = "balanced")
end

# plot regressed
if run_reg
  plot_solution(l96, plt, sol_reg, k = k, label = "regressed")
end

# plot online
if run_onl
  plot_solution(l96, plt, sol_onl, k = k, label = "online")
end

plt.legend()
plt.show()


