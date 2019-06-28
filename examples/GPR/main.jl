#!/usr/bin/julia --

import NPZ
import PyPlot
const plt = PyPlot

include("../../src/GPR.jl")

const gpr_data = NPZ.npzread("../../test/GPR/data/data_points.npy")

################################################################################
# main section #################################################################
################################################################################
# To avoid bloated and annoying warnings from ScikitLearn.jl, run this with
#
# julia --depwarn=no main.jl
#

# create an instance of GPR.Wrap with threshold set to -1
# by convention, negative threshold means use all data, i.e. do not subsample
gprw = GPR.Wrap(thrsh = -1)
#gprw = GPR.Wrap()

GPR.set_data!(gprw, gpr_data)

GPR.subsample!(gprw) # this form is unnecessary, `learn!` will do it anyway
#GPR.subsample!(gprw, 500) # ignore `gprw.thrsh` and subsample 500 points
#GPR.subsample!(gprw, indices = 1:10) # subsample points with indices `indices`

GPR.learn!(gprw) # fit GPR with "Const * RBF + White" kernel
#GPR.learn!(gprw, noise = 1e-8) # `noise` is *non-optimized* additional noise
#GPR.learn!(gprw, kernel = "matern") # "rbf" and "matern" are supported for now
#GPR.learn!(gprw, kernel = "matern", nu = 1) # Matern's parameter nu; 1.5 by def

mesh = minimum(gpr_data, dims=1)[1] : 0.01 : maximum(gpr_data, dims=1)[1]

mean, std = GPR.predict(gprw, mesh, return_std = true)
#mean = GPR.predict(gprw, mesh) # `return_std` is false by default

################################################################################
# plot section #################################################################
################################################################################
plt.plot(gpr_data[:,1], gpr_data[:,2], "r.", ms = 6, label = "Data points")
plt.plot(mesh, mean, "k", lw = 2.5, label = "GPR mean")
plt.fill_between(mesh, mean - 2*std, mean + 2*std, alpha = 0.4, zorder = 10,
                 color = "k", label = "95% interval")

plt.legend()
plt.show()


