"""
Simple example generating and using synthetic data
"""

using JLD, Random, PyPlot, PyCall #load packages
rc("font", family="sans-serif", size = 14)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]

include(joinpath(@__DIR__, "src/", "fit_gplvm.jl")) #fits the model
include(joinpath(@__DIR__, "src/", "fitting_data.jl")) #generates synthetic data
include(joinpath(@__DIR__, "src/", "initialize_parameters.jl")) #says on the tin
include(joinpath(@__DIR__, "analysis_code/", "alignment_functions.jl")) #aligns the result
include(joinpath(@__DIR__, "analysis_code/", "calc_LL.jl")) #compute importancve-weighted log likelihood
Random.seed!(9310906)
Plot = true #decide whether to plot stuff

##pick a model to fit
manifold = "1-torus" #generate data on the circle
fit_manifold = "1-torus" #fit data on the circle (model recovery)
fname = manifold*"_"*fit_manifold*"_test"
println("\n\n", " fitting ", fit_manifold, " to data generated from ", manifold)

##initialize parameters
N, T = 35, 200 #number of neurons and conditions
mind = 15 #number of inducing points
m = 100 #number of mc samples per update step
nmax = 500 #maximum number of training steps
minH = 50 #number of 'warm start' steps where the entropy term is ignored and variational Σ fixed
kmax = 3 #ReLie kmax (consider (2kmax+1) equivalent datapoints on the algebra tfor entropy calculations)
thresh = 0.01 #stopping condition
comb = true #link parameters across neurons for quicker computation

##generate data and initialize parameters
n, metric, ntilde = get_params(manifold);  #get some info about our model
#preferred orientations (xs), latent states (ps), signals variances (alphas),
#offsets (betas), length scales (ls), noise variances (sigs), scalings (gammas)
xs, ps, Y, alphas, betas, ls, sigs, gammas = gen_data(N, T, n, metric = metric, cont = true) #draw data from true manifold

n, metric, ntilde = get_params(fit_manifold) #get some info about the model we want to fit
μs, Σs, alphas0, ls0, sigs0, us, dfunc = initialize_parameters(fit_manifold, Y, n, T, mind, comb = comb) #initialize parameters
alpha = 0.075 #Adam learning rate

##plot the raw data
if Plot
    figure(figsize = (6, 4))
    argsx = sortperm(xs[:, 1])
    imshow(Y[argsx, :], cmap = "Greys", aspect = "auto") #sort by preferred direction
    xlabel("time"); ylabel("neurons")
    savefig("figures/example/"*fname*"_raw_data.png", bbox_inches = "tight")
    close()
end

##fit the model
Lfit, θs = fit_gplvm(Y, μs, Σs, alphas0, ls0, sigs0, us, n, dfunc, m=m,
                        nmax = nmax, alpha=alpha, kmax = kmax, minH = minH, thresh = thresh, comb = comb)

#compute log likelihood
println("computing likelihood")
LL = calc_LL(θs, Y, n, dfunc, m = 1000, comb = comb)
println("final log likelihood = ", LL)

##plot the result
if Plot
    μs = ( reduce(hcat, θs[1]) .+ 2*pi ) .% (2*pi)
    σs = sqrt.(reduce(hcat, θs[2]))
    #align result
    alpha, offset = align_theta(ps[:], μs[:], Print = false) #sign and rotation
    plotps = (alpha*ps[:] .+ offset .+ 2*pi) .% (2*pi) #rotate as appropriate
    figure(figsize = (4, 4))
    errorbar(plotps, μs[:], yerr = σs[:], color = "b", fmt = "o", markersize = 2) #plot with error bars
    xlabel("true latent", labelpad = -15); ylabel("inferred latent", labelpad = -20)
    xticks([0; 2*pi], [L"0"; L"2 \pi"]); yticks([0; 2*pi], [L"0"; L"2 \pi"])
    xlim(0, 2*pi); ylim(0, 2*pi)
    savefig("figures/example/"*fname*"_inferred_latents.png", bbox_inches = "tight")
    close()
end
