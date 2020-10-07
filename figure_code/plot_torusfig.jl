
using PyPlot, JLD, LinearAlgebra, PyCall
include(joinpath(@__DIR__, "../analysis_code/", "alignment_functions.jl"))
include(joinpath(@__DIR__, "../src/", "utils.jl"))
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]

figsize = (3,3)

## load some data

d = load("results/example2-torus.jld")
θs = d["learned"]
true_params = d["true"]
xs, ps, Y, alphas, betas, ls, sigs, gammas = true_params
N, T = size(Y)
θcols = [0.5*[0.5*(sin(ps[1,i])+sin(ps[2,i])); cos(ps[1,i]); cos(ps[2,i])].+0.5 for i = 1:T] #define colormap
n = 2


ext = ".pdf"
base = "figures/neurips/2-torus/2-torus_"

##align and plot alignment
thetas = ( [1; -1] .* reduce(hcat, θs[1]) .+ 10*2*pi .+ [pi; 0] ) .% (2*pi) #project onto manifold
alpha1, alpha2, offset1, offset2 = align_torus(ps, thetas) #align the degrees of freedom
plotps = ([alpha1; alpha2] .* ps .+ [offset1; offset2] .+ 10*2*pi ) .% (2*pi) #align coordinate systems

figure(figsize = figsize)
scatter(plotps[1, :], plotps[2, :], c = θcols, marker = "o", s = 15)
scatter(thetas[1, :], thetas[2, :], c = θcols, marker = "x")
xlabel(L"\theta_1", labelpad = -12); ylabel(L"\theta_2", labelpad = -10, rotation = 0)
box(true)
xlim(0, 2*pi); ylim(0, 2*pi)
xticks([0, 2*pi], [L"0", L"2 \pi"])
yticks([0, 2*pi], [L"0", L"2 \pi"])
savefig(base*"2d_scatter"*ext, bbox_inches = "tight")
close()

## load tuning curve data
for n = 1:10
    println(n)
    d = load("figures/2-torus/example_tuning"*string(n)*".jld")
    posterior = d["posterior"]
    truevals = d["true"]
    ps_gp = d["ps_gp"]

    Nb = Int(sqrt(size(ps_gp)[2]))
    truevals_imshow = Nb/(2*pi) * truevals #put in imshow coordinates
    truevals_imshow[truevals_imshow .> Nb-(1.5)] .-= Nb
    truevals_imshow[truevals_imshow .< -0.5] .+= Nb

    xvals = reshape(ps_gp[1,:], Nb, Nb)
    yvals = reshape(ps_gp[2,:], Nb, Nb)
    acts = reshape(mean(posterior, dims = 1), Nb, Nb)

    ## plot 2d tuning curve
    figure(figsize = (figsize[1]/2, figsize[2]/2))
    imshow(acts, cmap = "Reds")
    plot(truevals_imshow[:, 1], truevals_imshow[:, 2], "ko", markersize = 1)
    xticks([-0.5, Nb-0.5], [L"0", L"2 \pi"])
    yticks([-0.5, Nb-0.5], [L"2 \pi", L"0"])
    xlabel(L"\theta_1", labelpad = -12); ylabel(L"\theta_2", labelpad = -10, rotation = 0)
    box(true)
    savefig("figures/neurips/2-torus/tuning_curves/example_tuning"*string(n)*ext, bbox_inches = "tight")
    close()

    ##plot 3d tuning curve
    tcols = mean(posterior, dims = 1)[:]
    tcols = tcols .- minimum(tcols)
    tcols = tcols / maximum(tcols)
    tcols = 0.05 .+ tcols*0.9
    #cmap = get_cmap("coolwarm")
    cmap = get_cmap("Reds")
    tcols = cmap.(tcols)
    tcols = reshape(tcols, Nb, Nb)
    #maxes =

    xvalsp = [xvals xvals[:, 1]]; xvalsp = [xvalsp; xvalsp[1,:]']
    yvalsp = [yvals yvals[:, 1]]; yvalsp = [yvalsp; yvalsp[1,:]']
    plot_solid_torus(xvalsp, yvalsp, tcols, fname = "figures/neurips/2-torus/tuning_curves/projected_solid_tuning"*string(n), azim = 100, ext=ext,
                    figsize = figsize)
end


## plot CV error
d = load("results/cv/2-torus_cv_comparison.jld")
allerrs = d["allerrs"]
ELBOs = d["ELBOs"]
LLs = -d["LLs"]
tLLs = -d["tLLs"]
labels = ["MSE", "ELBO", "NLL", "NLL"]
names = ["MSE", "ELBO", "NLL", "tNLL"]
niters = size(ELBOs)[1]
n = 1
ticks = [L"\mathbb{R}^2"; L"T^2"]

data1 = (allerrs[1:niters, :] + allerrs[(niters+1):end, :])/2
data2 = ELBOs
data3 = LLs


cols = ["k" for i = 1:niters]
shift = 5*niters
xs1 = 1:niters
xs2 = (1:niters) .+ shift
ticksy = [[[0.05; 0.07], [L"0.05"; L"0.07"]], [[-21; -9], [L"-21"; L"-9"]],
            [[21; 9], [L"21", L"9"]], [[15; 34], [L"15", L"34"]]]
ypads = [-25, -25, -20, -20]

for (i, data) = enumerate([data1, data2, data3, tLLs])
    figure(figsize = (1.4, 3))
    for j = 1:niters
        y2, y1 = data[j, 1], data[j, 2]
        errorbar(xs1[j], [y1], yerr = [0], fmt = cols[i]*"o", capsize = 3)
        errorbar(xs2[j], [y2], yerr = [0], fmt = cols[i]*"o", capsize = 3)
        plot([xs1[j]; xs2[j]], [y1; y2], cols[i]*"--")
    end
    xticks([niters/2; shift+niters/2], ticks)
    yticks(ticksy[i][1], ticksy[i][2])
    ylabel(labels[i], labelpad = ypads[i])
    box(false)
    plot(zeros(2).-niters, ticksy[i][1], "k-")
    tick_params(axis="both", length = 0)
    tick_params(axis = "y", pad = -3)
    tick_params(axis = "x", pad = -5)
    for tick = ticksy[i][1] plot([-niters; -niters*1.5], [tick; tick], "k-") end
    savefig(base*"comparison_"*names[i]*ext, bbox_inches = "tight")
    close()
end
