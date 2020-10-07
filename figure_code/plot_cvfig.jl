using PyPlot, JLD, LinearAlgebra, PyCall, Statistics
include(joinpath(@__DIR__, "../src/", "utils.jl"))
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]
ext = ".pdf"


#%% plot error trajectories

fnames = ["results/cv/2-torus_cv_comparison_3fold.jld",
            "results/cv/SO3_cv_comparison_3fold.jld",
            "results/cv/3-sphere_cv_comparison_3fold.jld"]

#extract cv data
vals = [-load(fname)["tLLs"] for fname = fnames]

cols = ["b", "c", "g"]
alphas = [0.3, 0.5, 0.4]
Norm = true #log likelihood ratios

#plot in order T2, SO3, S3 for all files
inds = [[1,2,3], [3,1,2], [2,3,1]]
figure(figsize = (2.5, 2.5))
plot([], [], "k--", alpha = 0.3)
plot([], [], "k-")
legend(["dataset", "mean"], frameon = false, prop=Dict([("size", 12)]))

for j = 1:3
    for i = 1:size(vals[j])[1]

        plotval = vals[j][i, inds[j]]
        println(j, " ", i, " ", plotval)
        if Norm plotval .-= minimum(plotval) end
        plot(1:3, plotval, cols[j]*":", alpha = alphas[j])
    end
    plotval = mean(vals[j], dims = 1)[inds[j]]
    if Norm plotval .-= minimum(plotval) end
    plot(1:3, plotval, cols[j]*"-")
end
xticks([])

#add labels
for (i, lab) = enumerate([L"T^2", L"SO(3)", L"S^3"])
    text(i, -0.75, lab, color = cols[i], horizontalalignment="center", verticalalignment="top")
end

#add some labels etc.
ylabel("NLL", labelpad = -20)
ticksy = [10; 35]
if Norm ticksy = [0; 15]; ylabel("NLL - min(NLL)") end
yticks(ticksy)
box(false)
plot(zeros(2).+0.7, ticksy, "k-")
tick_params(axis="both", length = 0)
tick_params(axis = "y", pad = -3)
tick_params(axis = "x", pad = -5)
for tick = ticksy plot([0.6; 0.7], [tick; tick], "k-") end

savefig("figures/neurips/3fold/3fold_comparison.pdf", bbox_inches = "tight")
close()
