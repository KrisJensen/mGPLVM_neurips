"""
example simulation on SO(3)
"""

using Random, JLD
Random.seed!(9112705)
include(joinpath(@__DIR__, "../src/", "fit_gplvm.jl"))
include(joinpath(@__DIR__, "../src/", "fitting_data.jl"))
include(joinpath(@__DIR__, "../src/", "initialize_parameters.jl"))

manifold = "SO3"
fit_manifold = "SO3" #manifold to fit
println("\n\n", " fitting ", fit_manifold, " to data generated from ", manifold)

N, T, mind, m, nmax, minH = 300, 600, 30, 100, 900, 600
if fit_manifold == "SO3" kmax = 5 else kmax = 3 end
alpha = 0.02

n, metric, ntilde = get_params(manifold);  #data generation
xs, ps, Y, alphas, betas, ls, sigs, gammas = gen_data(N, T, n, metric = metric, cont = false, uncertainty = "low") #draw data from true manifold

true_params = [xs, ps, Y, alphas, betas, ls, sigs, gammas]
n, metric, ntilde = get_params(fit_manifold) #data fitting
μs, Σs, alphas0, ls0, sigs0, us, dfunc = initialize_parameters(fit_manifold, Y, n, T, mind) #initialize things
ls0 = [mean(ls0)]; sigs0 = [mean(sigs0)]; alphas0 = [mean(alphas0)]; us = [us[1]] #fix parameters

#run fitting procedure
Lfit, θs = fit_gplvm(Y, μs, Σs, alphas0, ls0, sigs0, us, n, dfunc, m=m, nmax = nmax, alpha=alpha,
                    kmax = kmax, comb = true, minH = minH, thresh=0)
if length(Lfit) == 1
    save("results/example"*fit_manifold*".jld", "learned", θs, "true", true_params, "Lfit", Lfit, "argsx", argsx)
end
