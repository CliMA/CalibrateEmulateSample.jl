module EnsembleKalmanInversion

export EKI

using Distributions, LinearAlgebra

struct EKI
    """
    truth[k] is the measurement of the kth output
    """
    truth::Vector{Float64}

    """
    cov[k1,k2] is the covariance of the (k1,k2) outputs
    """
    cov::Matrix{Float64}

    """
    u[i][k,j] is the kth parameter of the jth ensemble for the ith iteration
    """
    u::Vector{Matrix{Float64}}

    """
    size of the ensemble
    """
    J::Int
    
    """
    g[i][k,j] is the kth output of the jth ensemble for the ith iteration
    """
    g::Vector{Matrix{Float64}}
    
    """
    err[i] is the error of the ith iteration
    """
    err::Vector{Float64}   
end

function EKI(truth::Vector, cov::Matrix, u_init::Matrix)
    J = size(u_init,2)
    g = Matrix{Float64}[]
    err = Float64[]
    EKI(truth, cov, [u_init], J, g, err)
end

function get_u(eki)
    return mean(eki.u[end], dims=2)
end

function get_g(eki)
    return mean(eki.g[end], dims=2)
end


function residual(eki)
    g = get_g(eki)
    diff = g - eki.truth
    err = dot(diff,  eki.cov \ diff )
    push!(eki.err, err)
    return err
end


function update!(eki::EKI, g)

    u = eki.u[end]
    g_t = eki.truth
    cov = eki.cov
    J = eki.J

    us = size(u,1) # parameters x particles
    ps = size(g,1) # data x particles
    
    u_bar = zeros(us)
    p_bar = zeros(ps)
    c_up = zeros((us, ps))
    c_pp = zeros((ps, ps))

    for j in 1:J
        u_hat = u[:,j]
        p_hat = g[:,j]

        u_bar .+= u_hat
        p_bar .+= p_hat

        c_up .+= [u_hat[a] * p_hat[b] for a=1:us, b=1:ps]
        c_pp .+= [p_hat[a] * p_hat[b] for a=1:ps, b=1:ps]
    end

    u_bar = u_bar / J
    p_bar = p_bar / J
    c_up = c_up / J - [u_bar[a] * p_bar[b] for a=1:us, b=1:ps]
    c_pp = c_pp / J - [p_bar[a] * p_bar[b] for a=1:ps, b=1:ps]

    noise = rand(MvNormal(zeros(ps), cov), J)
    y = g_t .+ noise
    tmp = (c_pp + cov) \ (y - g)
    u += c_up * tmp

    push!(eki.u, u)
    push!(eki.g, g)
end

end # module
