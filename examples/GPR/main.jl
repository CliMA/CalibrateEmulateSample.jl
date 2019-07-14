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

mesh, mean, std = GPR.mmstd(gprw) # equispaced mesh; mean and std at mesh points
#mesh, mean, std = GPR.mmstd(gprw, mesh_n = 11) # by default, `mesh_n` is 1001

#mean, std = GPR.predict(gprw, mesh, return_std = true) # your own mesh
#mean = GPR.predict(gprw, mesh) # `return_std` is false by default

################################################################################
# plot section #################################################################
################################################################################
#GPR.plot_fit(gprw, plt, plot_95 = true) # by default, `plot_95` is false

# no legend by default, but you can specify yours in the following order:
# data points, subsample, mean, 95% interval
GPR.plot_fit(gprw, plt, plot_95 = true,
             label = ["Points", "Training", "Mean", "95% interval"])

plt.legend()
plt.show()


