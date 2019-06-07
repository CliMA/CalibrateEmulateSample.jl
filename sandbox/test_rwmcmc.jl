include("./rwmcmc.jl")

using Plots
#Here we will test convergence
#to a normal distribution with mean zero and variance one

#define negative log likelihood for ρ(θ) = exp ( - 0.5 θ^2 ) / sqrt(2π)
ρ(θ) = exp(- 0.5 * θ[1]^2) / sqrt(2π)
nll(θ) = - log(ρ(θ))

#initial value for θ, an array
init_𝚯 = [0.0]

#define perturbation function as a gaussian
perturb(θ) = θ .+ randn()

#error_scale is 1 in this case
error_scale = 1.0

#define the number of timesteps
nt = 10^4

#choose an output file name
filename = "rwmcmc_test"

#choose output frequency
freq = 1000

#run rwMCMC
Random.seed!(1234) #for reproducibility
markov_chain_with_save(nll, init_𝚯, perturb, error_scale, nt, filename, freq)

#load the file that was saved, this will load "param" and "error"
@load "rwmcmc_test"

#some actually good norms
index_vals = 1:(nt+1)
index = 1
println("The error in the mean is ")
println(mean(param[index,index_vals]))
println("The error in the standard deviation is ")
println(std(param[index,index_vals])-1.0)

#plot histogram and compare with analytic pdf: eyeball norm
index_vals = 1:(nt+1)
index = 1
min_p1 = minimum(param[index, index_vals])
max_p1 = maximum(param[index, index_vals])
dsp1 = (max_p1 - min_p1) / 100
bins = collect(min_p1:dsp1:max_p1)
p1 = histogram(param[index,index_vals], normalize=true, bins = bins, xlabel="parameters", ylabel="pdf", fillcolor = :lightgray, label = "MCMC result")
#overlay exact solution as well
θ = collect(-4:0.1:4)
plot!(θ, ρ.(θ), color="red", label = "exact pdf", lw = 4)
display(p1)
sleep(10) #for running from terminal
