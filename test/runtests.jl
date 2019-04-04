using Test
using EnsembleKalmanInversion, Distributions

let
    iter_max = 5
    J = 100
    n = 5
    r = 0.1
    # r = 0.0

    A = rand(Normal(0, 2), (n,n))
    u_t = rand(Normal(0, 3), n)
    g_t = A * u_t

    cov = rand(Normal(0, 2), (n,n))
    cov = cov' * cov
    cov *= r^2

    u_init = rand(Normal(0, 2), (n,J))
    g_ens = A * u_init

    eki = EKI(g_t, cov, u_init)

    iter = 0
    while iter < iter_max
        g_ens = A * eki.u[end]
        EnsembleKalmanInversion.update!(eki, g_ens)
        if EnsembleKalmanInversion.residual(eki) < 0.01
            break
        end
        iter += 1
    end

    @test u_t ≈ vec(EnsembleKalmanInversion.get_u(eki)) atol=0.01
end
