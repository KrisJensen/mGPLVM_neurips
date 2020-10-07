"""
example simulations on a fly
"""

using JLD, DelimitedFiles, Random
include(joinpath(@__DIR__, "../src/", "fit_gplvm.jl"))
include(joinpath(@__DIR__, "../src/", "fitting_data.jl"))
include(joinpath(@__DIR__, "../src/", "initialize_parameters.jl"))

Random.seed!(11073005)

class = "light"
fname = "fly1_trial1_light"

fit_manifold = "1-torus" #manifold to fit
mind, m, nmax, minH = 15, 100, 300, 100
kmax = 3

##load data
Y = readdlm("fly_data/"*fname*"_Y.tsv")
xs = readdlm("fly_data/"*fname*"_x.tsv")'
true_params = [xs, Y] #note that the xs are actually what we call ps (true latents; here head direction)
N, T = size(Y)

n, metric, ntilde = get_params(fit_manifold);
println("\n\n fitting ", fit_manifold, " to ", fname)
fname = fname*"_"*fit_manifold

##initialize parameters
μs, Σs, alphas0, ls0, sigs0, us, dfunc = initialize_parameters(fit_manifold, Y, n, T, mind) #initialize things
alpha = 0.4e-1 #learning rate

##fit model
Lfit, θs = fit_gplvm(Y, μs, Σs, alphas0, ls0, sigs0, us, n, dfunc, m=m, nmax = nmax, alpha=alpha, kmax = kmax, minH = minH, thresh = 0)
if length(Lfit) == 1
    #save result provided things didn't crash
    save("results/fly/"*fname*".jld", "learned", θs, "true", true_params, "Lfit", Lfit)
end
