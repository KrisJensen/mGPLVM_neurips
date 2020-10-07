using JLD, PyPlot, MultivariateStats, PyCall, LinearAlgebra
include(joinpath(@__DIR__, "../analysis_code/", "alignment_functions.jl"))
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]
pypatch = pyimport("matplotlib.patches")


class = "light"
fname = "fly1_trial1_light"
basef = fname
figsize = (3.8,3.8)

#load ARD result
d = load("results/ard_N50_2000.jld")
θs = d["learned"]
true_params = d["true"]
Lfits = d["Lfit"]
ps = true_params[1]
Y = true_params[2]
N, T = size(Y)
mus = (reduce(hcat, θs[1]) .+ 20*pi ) .% (2*pi)

#plot variational distributions
figure(figsize = figsize)
for i = 1:T
    global ells
    e = pypatch.Ellipse((mus[1, i], mus[2, i]), sqrt(θs[2][i][1,1]), sqrt(θs[2][i][2,2]), color = "k", alpha = 0.5, fill = true, visible = true)
    gca().add_artist(e)
end
xlim(0, 2*pi); ylim(0, 2*pi)
xticks([0; 2*pi], [L"0", L"2\pi"])
yticks([0; 2*pi], [L"0", L"2\pi"])
xlabel(L"\theta_1", labelpad = -15)
ylabel(L"\theta_2", labelpad = -15, rotation = 0)
savefig("figures/ard/ard_syn_qs.pdf", bbox_inches = "tight")
close()

#plot length scales
niters = size(Y)[1]
shift = 5*niters
xs1 = 1:niters
xs2 = (1:niters) .+ shift
ticksy = [0; 4; 8; 12; 16]
figure(figsize = (2,4))
for i = 1:N
    y1, y2 = θs[4][i, 1], θs[4][i, 2]
    errorbar(xs1[i], y1, yerr = 0, fmt = "ko", capsize = 3)
    errorbar(xs2[i], y2, yerr = 0, fmt = "ko", capsize = 3)
    plot([xs1[i]; xs2[i]], [y1; y2], "k--")
end
xticks([niters/2; niters/2+shift], [L"\ell_1", L"\ell_2"])
yticks(ticksy, ticksy)
box(false)
plot(zeros(length(ticksy)).-niters, ticksy, "k-")
tick_params(axis="both", length = 0)
tick_params(axis = "y", pad = -3)
tick_params(axis = "x", pad = -5)
for tick = ticksy plot([-niters; -niters*1.5], [tick; tick], "k-") end
savefig("figures/ard/ard_syn_ls.pdf", bbox_inches = "tight")
close()

#plot each dimension against true latent
theta1ts = ps[:] #true latents
alpha, offset = align_theta(mus[1, :], theta1ts)
xvals1 = (alpha*mus[1, :] .+ offset .+2*pi) .% (2*pi) #align first dimension
alpha, offset = align_theta(mus[2, :], theta1ts)
xvals2 = (alpha*mus[2, :] .+ offset .+2*pi) .% (2*pi) #align second dimension
xmin, xmax = 0, 2*pi
figure(figsize = figsize)
scatter(theta1ts, xvals1, c = "k", s =12)
scatter(theta1ts, xvals2, c = "b", s =12)
xlabel(L"\theta_{true}", labelpad = -15); ylabel(L"\theta_{fit}", rotation = 0, labelpad = -10)
xlim(0, 2*pi); ylim(0, 2*pi)
xticks([0; 2*pi], [L"0", L"2\pi"])
yticks([0; 2*pi], [L"0", L"2\pi"])
legend([L"\theta_1", L"\theta_2"], loc = "lower right")
savefig("figures/ard/ard_syn_thetas.pdf", bbox_inches = "tight")
close()
