#Random Walk Markov Chain Monte Carlo (rwmcmc) functions
#a starting point for thinking about abstractions that may be useful

using JLD2
using BenchmarkTools
using Statistics
using Distributions
using Random
#use the package below for debugging
using Revise

"""
take_step(dε)

Description

    Determines the accept or reject criteria for rwmcmc

Input

    dε: difference of (scaled) negative log likelood functions

Output

    Boolean Value: True or False

"""
take_step(dε) = log(rand(Uniform(0, 1))) < dε



"""
torus(x, a, b)

Description

    Takes x ∈ ℝ and outputs x ∈ [a, b] in a periodic way.
    (unicode shortcut is bbR for ℝ and in for ∈)

    If a partical is moving to the right then it will pop from b to the point a


Input

    x: a real valued scalar
    a: left endpoint of interval
    b: right endpoint of interval

Output

    a value in the interval [a,b]

"""
torus(x, a, b) = (((x-a)/(b-a))%1 - 0.5 * (sign((x-a)/(b-a)) - 1) )*(b-a) + a


"""
torus(x) = torus(x, 0, 1)

Description

    see torus(x, a, b)

"""
torus(x) = torus(x, 0, 1)


"""
markov_link(nll, param, ε, ε_scale, perturb)

Description

    Takes a single step in the random walk markov chain monte carlo algorithm

Input

    nll: The negative log-likelihood function. In the absence of priors this becomes a loss function

    𝚯: Array of vectors at the current iteration (unicode is bfTheta )

    ε: value of nll(𝚯)

    ε_scale: an overall scale for the output of nll

    perturb: a function that performs a perturbation of 𝚯

Return: new_𝚯, new_ε, test_𝚯, test_ε

    new_𝚯: The value of the new accepted 𝚯

    new_ε: value of nll(new_𝚯)

    test_𝚯: The 𝚯 from the "proposal step". Was either rejected or accepted

    test_ε: value of nll(test_𝚯)


"""
function markov_link(nll, 𝚯, ε, ε_scale, perturb)
    test_𝚯 = perturb(𝚯)
    test_ε = nll(test_𝚯)
    d_ε = (ε - test_ε) / ε_scale
    if take_step(d_ε)
        new_ε = test_ε
        new_𝚯 = test_𝚯
    else
        new_ε = ε
        new_𝚯 = 𝚯
    end
    return new_𝚯, new_ε, test_𝚯, test_ε
end


"""
markov_chain_with_save(nll, init_𝚯, perturb, ε_scale, nt, filename, freq)

Description

    A random walk of parameters that computes the posterior distribution

Input

    nll: The negative log-likelihood function. In the absence of priors this becomes a loss function

    init_𝚯: Array of initial parameter values (unicode is bfTheta )

    perturb: a function that indicates how to perform the random walk

    ε_scale: an overall error scale. shows up as nll(𝚯) / ε_scale

    nt: number of markov chain monte carlo steps

    perturb: a function that performs a perturbation of 𝚯

    filename: name for output file in JLD2 format

    freq: how often to save output (in terms of iterations)

Return: param, ε

    param: The matrix of accepted parameters in the random walk

    ε: The array of errors associated with each step in param chain

"""
function markov_chain_with_save(nll, init_𝚯, perturb, ε_scale, nt, filename, freq)
    param = ones(length(init_𝚯),nt+1)
    @views @. param[:,1] = init_𝚯
    test_param = deepcopy(param)
    ε = ones(nt+1) .* 10^6
    test_ε = deepcopy(ε)
    ε[1] = nll(init_𝚯)
    for i in 1:nt
        new_param, new_ε, proposal_param, proposal_ε = markov_link(nll, param[:,i], ε[i], ε_scale, perturb)
        @views @. param[:,i+1] = new_param
        ε[i+1] = new_ε
        @views @. test_param[:,i+1] = proposal_param
        test_ε[i+1] = proposal_ε
        if i%freq==0
            println("saving index "*string(i))
            @save filename ε param
        end
    end
    return param, ε
end
