using PyPlot, JLD, LinearAlgebra, InvertedIndices, Random, PyCall, DelimitedFiles
include(joinpath(@__DIR__, "../src/", "utils.jl"))
include(joinpath(@__DIR__, "../analysis_code/", "alignment_functions.jl"))
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]

figsize = (2.5,2.5)
base = "figures/neurips/SO3/SO3_"

## load some data
d = load("results/exampleSO3.jld")
θs = d["learned"]
true_params = d["true"]
xs, ps, Y, alphas, betas, ls, sigs, gammas = true_params
N, T = size(Y)
n = size(θs[2][1])[1]

mus = readdlm("results/so3_aligned.csv")'
newps = ps

vs1 = logmap_so3(newps')
vs2 = logmap_so3(mus')

##plot magnitudes against one another
mags1 = sqrt.(sum(vs1.^2, dims = 2)[:])
mags2 = sqrt.(sum(vs2.^2, dims = 2)[:])
figure(figsize = figsize)
plotmags1 = copy(mags1); plotmags1[vs1[:, 1] .< 0] = pi .- plotmags1[vs1[:, 1] .< 0]
plotmags2 = copy(mags2); plotmags2[vs2[:, 1] .< 0] = pi .- plotmags2[vs2[:, 1] .< 0]
scatter(2*plotmags1, 2*plotmags2, c = "k", s = 5, marker = "o", alpha = 0.1)
xlabel(L"\theta_{true}", labelpad = -12)
ylabel(L"\theta_{fit}", labelpad = -18)
xticks([0, 2*pi],[L"0", L"2 \pi"])
yticks([0, 2*pi],[L"0", L"2 \pi"])
box(false)
plot([0;0], [0; 2*pi], "k-"); plot([0;2*pi], [0; 0], "k-")
tick_params(axis="both", length = 0)
tick_params(axis = "both", pad = -3)
tickl = 0.1
for tick = [0; 2*pi] plot([0; -tickl], [tick; tick], "k-");  plot([tick; tick], [0; -tickl], "k-") end
savefig(base*"magnitudes.pdf", bbox_inches = "tight")
close()

## plot us
us1 = vs1 ./ mags1
us2 = vs2 ./ mags2

us1 = sign.(us1[:, 1]) .* us1
rs1 = sqrt.(us1[:,1].^2+us1[:,2].^2)
ϕs1 = (sign.(us1[:,2]).*acos.(us1[:, 1]./rs1) .+ pi/2) #0:2pi
θs1 = acos.(us1[:, 3]) #0:2pi
us2 = sign.(us2[:, 1]) .* us2
rs2 = sqrt.(us2[:,1].^2+us2[:,2].^2)
ϕs2 = (sign.(us2[:,2]).*acos.(us2[:, 1]./rs2) .+ pi/2) #0:2pi
θs2 = acos.(us2[:, 3]) #0:2pi

colsh = [0.5*[0.5*(sin(2 * ϕs1[i])+sin(2*θs1[i])); cos( 2 * ϕs1[i] ); cos(2*θs1[i])].+0.5 for i = 1:length(mags1)]
#color scheme is smooth in true coordinate system
figure(figsize = figsize)
scatter3D(us2[:, 1], us2[:, 2], us2[:,3], marker = "x", c = colsh, s = 30)
xlim(-1.05, 1.05); ylim(-1.05, 1.05); zlim(-1.05, 1.05)
ax=gca(); ax.set_axis_off()
gca().view_init(elev=0, azim=0)
savefig(base*"us.pdf", bbox_inches = "tight")
close()

## plot true and ref in polar coordinates
us1 = sign.(us1[:, 1]) .* us1
rs1 = sqrt.(us1[:,1].^2+us1[:,2].^2)
ϕs1 = (sign.(us1[:,2]).*acos.(us1[:, 1]./rs1) .+ pi/2) #0:2pi
θs1 = acos.(us1[:, 3]) #0:2pi
us2 = sign.(us2[:, 1]) .* us2
rs2 = sqrt.(us2[:,1].^2+us2[:,2].^2)
ϕs2 = (sign.(us2[:,2]).*acos.(us2[:, 1]./rs2) .+ pi/2) #0:2pi
θs2 = acos.(us2[:, 3]) #0:2pi
figure(figsize = (2.3, 2.3))
scatter(ϕs2, θs2, marker = "x", c = colsh, s = 12, linewidths = 0.2)
scatter(ϕs1, θs1, marker = "o", c = colsh, s = 5)
xlim(0, 1*pi); ylim(0, pi)
xlabel(L"\phi", labelpad = -12); ylabel(L"\theta", labelpad = -8, rotation = 0)
box(true)
xlim(0, 1*pi); ylim(0, 1*pi)
xticks([0, 1*pi], [L"0", L"\pi"])
yticks([0, 1*pi], [L"0", L"\pi"])
savefig(base*"us_polar.pdf", bbox_inches = "tight")
close()


## plot CV stuff

d = load("results/cv/SO3_cv_comparison.jld")
allerrs = d["allerrs"]
ELBOs = d["ELBOs"]
LLs = -d["LLs"]
tLLs = -d["tLLs"]
labels = ["MSE", "ELBO", "NLL", "NLL"]
names = ["MSE", "ELBO", "NLL", "tNLL"]
niters = size(ELBOs)[1]
n = 1
ticks = [L"\mathbb{R}^3"; L"SO(3)"]

data1 = (allerrs[1:niters, :] + allerrs[(niters+1):end, :])/2
data2 = ELBOs
data3 = LLs

cols = ["k" for i = 1:niters]
shift = 5*niters
xs1 = 1:niters
xs2 = (1:niters) .+ shift
ticksy = [[[0.05, 0.07], [L"0.05", L"0.07"]], [[-12, -29], [L"-12", L"-29"]],
            [[10, 29], [L"10", L"29"]], [[20, 44], [L"20", L"44"]]]
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
    savefig(base*"comparison_"*names[i]*".pdf", bbox_inches = "tight")
    close()
end
