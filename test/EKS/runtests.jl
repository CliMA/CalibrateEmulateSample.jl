using Test
using CalibrateEmulateSample
const CES = CalibrateEmulateSample
using LinearAlgebra, Distributions

@testset "EKS" begin

    for FT in [Float32, Float64]
        Σ = FT(100)^2*Matrix{FT}(I,2,2)
        μ = FT[3,5]
        prior = MvNormal(μ, Σ)

        A = hcat(ones(FT,10), FT[i == 1 ? FT(1) : FT(0) for i = 1:10])
        f(x) = A*x

        u_star = FT[-1.0,2.0]
        y_obs = A*u_star

        Γ = FT(0.1)*Matrix{FT}(I,10,10)

        prob = CES.CESProblem(prior, f, y_obs, CES.CovarianceSpace(Γ))

        J = 50
        θs = [rand(FT,2) for i in 1:J]

        for i = 1:20
            fθs = map(f, θs)
            θs = CES.eks_iter(prob, θs, fθs)
        end

        postΣ = inv(inv(Σ) + A'*inv(Γ)*A)
        postμ = postΣ*(inv(Σ)*μ + A'*inv(Γ)*y_obs)


        @test mean(θs) ≈ postμ atol=2e-1
        @test cov(θs) ≈ postΣ atol=5e-1
    end
end
