"""
code for analyzing the results of mGPLVMs fitted to fly data
generates latent trajectories and tuning curves
"""

using JLD, PyPlot, MultivariateStats, LinearAlgebra
include(joinpath(@__DIR__, "../src/", "translation_functions.jl"))
include("gplvm_inference.jl")
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)

fname = "fly1_trial1_light"
fit_manifolds = ["1-torus"] #fitted manifold

ds = [load("results/fly/"*fname*"_"*m*".jld") for m = fit_manifolds]
θs = [d["learned"] for d = ds]
θs[1][2] = [Σ + 1e-4*I for Σ = θs[1][2]]
true_params = [d["true"] for d = ds]
ps = [true_param[1] for true_param = true_params]
Ys = [true_param[2] for true_param = true_params]
ext = ".pdf"
base = "neurips/fly/"*fname

##plot raw data
figure(figsize = (6, 3))
imshow(reverse(Ys[1][:, :], dims = 1), cmap = "Greys", aspect = "auto")
xticks([], [])
yticks([], [])
xlabel("time")
ylabel("neurons")
savefig("figures/"*base*"Y"*ext, bbox_inches = "tight")
close()

##superimpose latent on data
vals = generate_trajectory(θs[1], [], type = "torus", figsize = (6,3), fname = base*"1-torus_all_", ext = ext)

##generate tuning curves
titles = [L"T^1", L"R^1"]
for (i, fit_manifold) = enumerate(["1-torus", "1-cube"])
    println("plotting tuning curves ", fit_manifold)
    d = load("results/fly/"*fname*"_"*fit_manifold*".jld") #load data
    θ=d["learned"]; true_params=d["true"]
    println("Lfit: ", d["Lfit"])
    Y = true_params[2]
    if fit_manifold == "1-torus" dfunc = dtorus_gp else dfunc = deuclid_gp end
    plot_fly_tunings(Y, θ, dfunc, figsize = (4,3), fname = base*"_"*fit_manifold*"_", ext = ext)
end
