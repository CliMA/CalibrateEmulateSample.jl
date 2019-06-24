using Test
using Solus
using LinearAlgebra, Distributions

@testset "NEKI" begin

    Σ = 100.0^2*Matrix{Float64}(I,2,2)
    μ = [3.0,5.0]
    prior = MvNormal(μ, Σ)
    
    A = hcat(ones(10), [i == 1 ? 1.0 : 0.0 for i = 1:10])
    f(x) = A*x

    u_star = [-1.0,2.0]
    y_obs = A*u_star
    
    Γ = 0.1*Matrix{Float64}(I,10,10)
    
    prob = Solus.SolusProblem(prior, f, y_obs, Solus.CovarianceSpace(Γ))

    J = 50
    θs = [rand(2) for i in 1:J]

    for i = 1:20
        fθs = map(f, θs)
        θs = Solus.neki_iter(prob, θs, fθs)
    end

    postΣ = inv(inv(Σ) + A'*inv(Γ)*A)
    postμ = postΣ*(inv(Σ)*μ + A'*inv(Γ)*y_obs)

    
    @test mean(θs) ≈ postμ atol=2e-1
    @test cov(θs) ≈ postΣ atol=5e-1
end
