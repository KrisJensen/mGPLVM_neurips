using JLD, PyPlot, MultivariateStats, LinearAlgebra, DelimitedFiles, PyCall
include(joinpath(@__DIR__, "../analysis_code/", "gplvm_inference.jl"))
include(joinpath(@__DIR__, "../analysis_code/", "alignment_functions.jl"))
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]

fname = "fly1_trial1_light"
fit_manifolds = ["1-torus"] #fitted manifold

ds = [load("results/fly/"*fname*"_"*m*".jld") for m = fit_manifolds]
θs = [d["learned"] for d = ds]
true_params = [d["true"] for d = ds]
ps = [true_param[1] for true_param = true_params]
Ys = [true_param[2] for true_param = true_params]
ext = ".pdf"
base = "neurips/fly/"*fname

##superimpose latent on data
saved = readdlm("figures/"*base*"1-torus_all_thetas.tsv") #posterior trajectory
vals = [saved[:, 3]'; saved[:, 2]'; saved[:, 4]']

#align superimposed data
means = vals[2, :]
targets = [m[1] for m = argmax(Ys[1], dims = 1)][:]/16*2*pi
alpha_s, offset_s = align_theta(means, targets) #sign and rotation
plotvals = (vals .* alpha_s .+ offset_s .+ 2*pi ) .% (2*pi)
plotvals[1, :] = plotvals[2, :] + (vals[1,:] - vals[2,:])
plotvals[3, :] = plotvals[2, :] + (vals[3,:] - vals[2,:])
plotvals = plotvals/(2*pi)*16 .- 0.5
yerrs = [plotvals[2, :]' - plotvals[1,:]'; plotvals[3, :]' - plotvals[2,:]']

#only plot every other timepoint to avoid things looking too crowded
plots = 2:2:500
xvals = (0:499)[plots]
yvals = copy(plotvals[2,plots])
err1, err2 = copy(plotvals[1,plots]), copy(plotvals[3, plots])
diffs = [0; abs.(yvals[2:end] - yvals[1:(length(xvals)-1)])]
yvals[diffs .> 8] .= NaN; err1[diffs .> 8] .= NaN; err2[diffs .> 8] .= NaN

#plot figure
figure(figsize = (4, 3.2))
imshow(Ys[1], cmap = "Greys", aspect = "auto")
plot(xvals, yvals, color = "orange", ls = "-", linewidth = 1)
fill_between(xvals, err1, err2, color = "orange", alpha = 0.4)
xticks([], [])
yticks([], [])
box(false)
xlabel("time", labelpad = -9)
ylabel("neurons")
savefig("figures/"*base*"Y_sup"*ext, bbox_inches = "tight")
close()


## plot CV error
d = load("results/fly_cv/"*fname*"_cv_comparison.jld")
allerrs = d["allerrs"]
ELBOs = d["ELBOs"]
LLs = -d["LLs"]
tLLs = -d["tLLs"]
labels = ["MSE", "ELBO", "NLL", "NLL"]
names = ["MSE", "ELBO", "NLL", "tNLL"]
niters = size(ELBOs)[1]
n = 1
ticks = [L"\mathbb{R}^1", L"T^1"]

data1 = (allerrs[1:niters, :] + allerrs[(niters+1):end, :])/2
data2 = ELBOs
data3 = LLs

cols = ["k" for i = 1:niters]
shift = 5*niters
xs1 = 1:niters
xs2 = (1:niters) .+ shift
ticksy = [[[0.025, 0.038], [L"0.025", L"0.038"]], [[3.0, 4.6], [L"3.0", L"4.6"]],
            [[-3.1, -4.7], [L"-3.1", L"-4.7"]], [[-1.8, -0.1], [L"-1.8", L"-0.1"]]]
ypads = [-25, -25, -25, -25]

for (i, data) = enumerate([data1, data2, data3, tLLs]) #MSE, ELBO, train LL, test LL
    figure(figsize = (1.4, 3))
    for j = 1:niters
        y1, y2 = data[j, 2], data[j, 1]
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
    savefig("figures/"*base*"_comparison_"*names[i]*ext, bbox_inches = "tight")
    close()
end

## plot tuning curves
#note: need to run analyze_flyfig.jl first
titles = [L"T^1", L"\mathbb{R}^1"]
for (i, fit_manifold) = enumerate(["1-torus", "1-cube"]) #torus and euclidean
    println("plotting tuning curves ", fit_manifold)
    d = load("figures/"*base*"_"*fit_manifold*"_tuning_posteriors.jld")
    bins = d["bins"]; posterior = d["posterior"]
    figure(figsize = (4,3))
    for i = 2:2:16
        vals = reduce(hcat, [quantile(posterior[i][:, b], [0.025; 0.5; 0.975]) for b = 1:length(bins)])
        for b = 1:length(bins)
            m, s = mean(posterior[i][:, b]), 2*std(posterior[i][:, b]) #pm 2std
            vals[2, b] = m; vals[1, b] = m-s; vals[3, b] = m+s
        end
        means, shades1, shades2 = vals[2,:], vals[1,:], vals[3,:]
        col = (argmax(means)+50) / (length(bins)+100)
        col = [0; col; 1-col]
        plot(bins[:], means, color = col, ls = "-")
        fill_between(bins[:], shades1, shades2, color = col, alpha = 0.2)
    end
    plot([minimum(bins); maximum(bins)], [0; 0], "k--")
    yticks([0; 1], [0; 1])
    if fit_manifold == "1-torus"
        xticks([0; pi; 2*pi], [L"-\pi", L"0", L"\pi"]); xlabel(L"\theta")
        ylabel("activity", labelpad = -10)
    else
        xticks([-2;0;2], [L"-2", L"0", L"2"]); xlabel(L"x")
    end
    ylim(-0.2, 1.05)
    xlim(minimum(bins), maximum(bins))
    savefig("figures/"*base*"_"*fit_manifold*"_tuning_posteriors"*ext, bbox_inches = "tight")
    close()
end
