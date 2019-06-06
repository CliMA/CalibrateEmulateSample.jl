using Parameters # lets you have defaults for fields

@with_kw mutable struct L96m
  #predictor::Function
    K::Int = 9
    J::Int = 8
    hx::Float32 = -0.8
    hy::Float32 = 1
    F::Float32 = 10
    eps::Float32 = 2^(-7)
end


"""
    lorenz96

Compute the right-hand side of the Lorenz 96 system.

Parameters:
 - `F`: forcing term: can be either a scalar or an array.
"""
function lorenz96(dxdt, x, p, t)
    n = length(x)
    for i in 1:n
        F = p.F isa AbstractArray ? p.F[i] : p.F            
        dxdt[i] = (-x[mod1(i-1,n)] * (x[mod1(i-2,n)] - x[mod1(i+1,n)]) -x[i] + F)
    end
end


"""
    lorenz96c

Compute the right-hand side of the complementary Lorenz 96 system (operates in the opposite direction).

Parameters:
 - `F`: forcing term: can be either a scalar or an array.
 - `eps`: scale separation term
"""
function lorenz96c(dydt, y, p, t)
    n = length(y)
    for j in 1:n
        F = p.F isa AbstractArray ? p.F[i] : p.F            
        dydt[j] = (-y[mod1(j+1,n)] * (y[mod1(j+2,n)] - y[mod1(j-1,n)]) -y[j] + F)/p.eps
    end
end

"""
    lorenz96multiscale

Compute the right-hand side of the multiscale Lorenz 96 system.

Parameters:
 - `K`: number of slow variables
 - `J`: number of fast variables per slow variable
 - `F`: forcing term for slow variables
 - `hx`: slow ? term
 - `hy`: fast ? term
 - `eps`: scale separation term
"""
function lorenz96multiscale(dudt, u, p, t)
    K = p.K
    J = p.J
    
    x = @view(u[1:K])
    dxdt = @view(dudt[1:K])

    Y = reshape(@view(u[K .+ (1:J*K)]), J, K)
    dYdt = reshape(@view(dudt[K .+ (1:J*K)]), J, K)

    Fx = p.F .+ p.hx/J .* vec(sum(Y,dims=1))
    lorenz96(dxdt, x, (F=Fx,), t)

    for k in 1:K
        y = @view(Y[:,k])        
        dydt = @view(dYdt[:,k])
        Fy = p.hy*x[k]
        lorenz96c(dydt, y, (F=Fy,eps=p.eps), t)
    end
end


function lorenz96func(dxdt, x, p, t)
    n = length(x)
    for i in 1:n
        dxdt[i] = (-x[mod1(i-1,n)] * (x[mod1(i-2,n)] - x[mod1(i+1,n)]) -x[i] + p.G(x[i]))
    end
end
