using JLD, PyPlot, MultivariateStats, LinearAlgebra, Statistics, PyCall
include(joinpath(@__DIR__, "../analysis_code/", "calc_LL.jl"))
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]

fname = "fly1_trial1_light"
base = "neurips/fly/"*fname*"_"
ext = ".pdf"
figsize = (3,3)
cfact = 0.3
lpad = 9

##define color schemes
d = load("results/fly/"*fname*"_1-torus.jld")
θ1ts = d["learned"]
Y = d["true"][2]
theta1s = (reduce(vcat, θ1ts[1])[:] .+ 20*pi ) .% (2*pi)
θcols = [[0, cfact*cos(c)+0.5, 0.5-cos(c)*cfact] for c = theta1s] #color according to angle
acts = mean(Y, dims = 1)[:]; acts = acts .- minimum(acts); acts = acts / maximum(acts) #calculate mean activity
rcols = [[0, r, 1-r] for r = acts] #color according to global activity


## fit direct product

d = load("results/fly/fly1_trial1_light_dp.jld")
θ2ds = d["learned"]
thetas = reduce(hcat, θ2ds[1]) .+ [0; 0]
thetas[1, :] = ( thetas[1,:] .+ 10*2*pi ) .% (2*pi) #project onto manifold

xmin, xmax = -2.5, 3.5
figure(figsize = figsize)
scatter(thetas[2, :], acts, c = "k", s =12)
xlabel(L"x_t^{(T^1 \times \mathbb{R}^1)}", labelpad = -18); ylabel(L"\bar{y}_t", rotation = 0, labelpad = -10)
#title(L"T^1 \times \mathbb{R}^1")
box(false)
xticks([xmin, xmax], [L"-2.5"; L"3.5"])
plot([xmin, xmin], [0; 1], "k-"); plot([xmin, xmax], [0; 0], "k-")
tick_params(axis="both", length = 0)
tick_params(axis = "both", pad = -3)
tickl = 0.05
for yval = [0;1] plot([xmin; xmin-tickl*(xmax-xmin)], [yval; yval], "k-") end
for xval = [xmin;xmax] plot([xval; xval], [0; 0-tickl], "k-") end
yticks([0, 1], [L"0", L"1"])
savefig("figures/"*base*"T1_R1_rs"*ext, bbox_inches = "tight")
close()

xmin, xmax = 0, 2*pi
figure(figsize = figsize)
scatter(thetas[1, :], theta1s, c = "k", s =12)
xlabel(L"\theta_t^{(T^1 \times \mathbb{R}^1)}", labelpad = -18); ylabel(L"\theta_t^{T^1}", rotation = 0, labelpad = -15)
box(false)
xticks([xmin, xmax], [L"0", L"2\pi"])
yticks([xmin, xmax], [L"0", L"2\pi"])
plot([0; 0], [0; 2*pi], "k-"); plot([0; 2*pi], [0; 0], "k-")
tick_params(axis="both", length = 0)
tick_params(axis = "both", pad = -3)
tickl = 0.05
for yval = [0;2*pi] plot([xmin; xmin-tickl*2*pi], [yval; yval], "k-") end
for xval = [xmin;xmax] plot([xval; xval], [0; 0-tickl*2*pi], "k-") end
savefig("figures/"*base*"T1_R1_thetas"*ext, bbox_inches = "tight")
close()
