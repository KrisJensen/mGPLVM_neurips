"""
various functions for doing posterior inference on a fitted gplvm model
"""

using Zygote, DelimitedFiles, PyPlot
include(joinpath(@__DIR__, "../src/", "utils.jl"))
include(joinpath(@__DIR__, "../src/", "gplvm_utils.jl"))
include(joinpath(@__DIR__, "../src/", "distance_functions_gp.jl"))
include(joinpath(@__DIR__, "../src/", "distance_functions.jl"))
rc("font", family="sans-serif", size = 12)
rc("pdf", fonttype = 42)

function calc_sparse_posterior(x, X, Y, us, sig, alpha, l, dfunc)
    #x are points of evaluation (nxTtest)
    #X are training inputs (nxT)
    #Y are training outputs (Tx1)
    #us are inducing points (nxm)
    #follows titsias 2009

    Kfu =  calc_K(X, us, alpha, l, dfunc)
    Kuu = calc_K(us, us, alpha, l, dfunc)

    #compute phi = N(mu, A)
    Σinv = Kuu + sig^(-2) * Kfu' * Kfu #mxm
    ChS = cholesky(Symmetric(Σinv+1e-6*I))
    temp = ChS.U \ (ChS.L \ Kfu') #mxT
    mu = sig^(-2) * Kuu * temp * Y #mx1
    temp = ChS.U \ (ChS.L \ Kuu)
    A = Kuu * temp

    #compute posterior mean and covariance
    Kxx = calc_K(x, x, alpha, l, dfunc)
    Kxu = calc_K(x, us, alpha, l, dfunc)
    ChK = cholesky(Kuu + 1e-6*I)

    temp = ChK.U \ (ChK.L \ mu)
    m = Kxu * temp #mean

    temp1 = ChK.U \ (ChK.L \ Kxu')
    temp2 = ChK.U \ (ChK.L \ A)
    K = Kxx - Kxu * temp1 + Kxu * temp2 * temp1 #covariance

    return m, K #mean and covariance matrix

end

function plot_fly_tunings(Y, θs, dfunc; figsize = (5, 3.5), fname = "./", ext = ".png")
    #compute tuning curves for fly models
    N, T = size(Y)
    n = size(θs[2][1])[1]
    if dfunc == dtorus_gp
        xlab = L"\theta"
        ticks = [[0; pi; 2*pi], [L"0"; L"\pi"; L"2\pi"]]
        bins = -0.02:0.02:2.01*pi
        lims = [0; 2*pi]
    else
        maxes = reduce(vcat, θs[1]) + 2*reduce(vcat, θs[2])
        mins = reduce(vcat, θs[1]) - 2*reduce(vcat, θs[2])
        #plot in the region covered by the latents
        bins = range(minimum(mins), maximum(maxes), length = 300)
        xlab = "x"
        ticks = [round(ceil(minimum(bins))); 0; round(floor(maximum(bins)))]
        ticks = [ticks, ticks]
        lims = [minimum(bins); maximum(bins)]
    end
    println(bins)
    ps_gp = Array(bins)'
    nsamps = 1000
    posterior = [zeros(nsamps, length(bins)) for i = 1:N]
    for j = 1:nsamps
        Xtildes = [randn(1, n) for i = 1:T]
        Xs = reduce(vcat, [calc_X(Xtildes[i], θs[1][i], θs[2][i]) for i = 1:T])'

        for ind = 1:N
            alphast, lst, sigst, us = θs[3][ind], θs[4][ind], θs[5][ind], θs[6][ind] #model parameters
            mus, covs = calc_sparse_posterior(ps_gp, Xs, Y[ind, :], us, sigst, alphast, lst, dfunc)
            samps = randn(1, length(mus))
            samps = samps * cholesky(Symmetric(covs)+1e-6*I).U .+ mus' #sample from posterior
            posterior[ind][j, :] = samps[:]
        end
    end

    cols = [[0, i/(N+1), 1-i/(N+1)] for i = 1:N]
    figure(figsize = figsize)
    for i = 1:N
        vals = reduce(hcat, [quantile(posterior[i][:, b], [0.025; 0.5; 0.975]) for b = 1:length(bins)])
        means, shades1, shades2 = vals[2,:], vals[1,:], vals[3,:]
        col = (argmax(means)+50) / (length(bins)+100)
        col = [0; col; 1-col]
        plot(bins[:], means, color = col, ls = "-")
        fill_between(bins[:], shades1, shades2, color = col, alpha = 0.2)
    end
    xlabel(xlab)
    yticks([], [])
    ylabel("activity")
    xticks(ticks[1], ticks[2])
    xlim(lims[1], lims[2])
    savefig("figures/"*fname*"tuning_posteriors"*ext, bbox_inches = "tight")
    close()
    save("figures/"*fname*"tuning_posteriors.jld", "bins", bins, "posterior", posterior)
end

function generate_trajectory(θs, plotps; figsize = (7, 3.5), shift = 0, type = "torus", fname = "./", ext = ".png", plotinds = "default")
    #code for generating latent trajectories from a fitted model

    T = length(θs[1]) #number of data points
    n = length(θs[1][1]) #number of latent dimensions

    if plotinds == "default" plotinds = 1:T end
    if occursin("euclid", type)
        bins = range(minimum(plotps)-0.1, maximum(plotps)+0.1, length = 300)
    else
        bins = 0.01:0.02:2*pi
    end
    qs = [0.025; 0.5; 0.975] #plot 95th confidence interval
    vals = zeros(3, length(plotinds))

    μs = [μ for μ = θs[1]]
    if shift != 0
        plotps = (plotps .+ shift .+ 2*pi) .% (2*pi)
        plotps = (plotps .+ shift .+ 2*pi) .% (2*pi)
        μs = [(μ .+ shift .+ 2*pi) .% (2*pi) for μ = θs[1]]
    end

    for (i, ind) = enumerate(plotinds)
        Xtildes = randn(10000, n); density = zeros(length(bins))
        Xs = calc_X(Xtildes, μs[ind], θs[2][ind])
        thetaXs = Xs[:]
        if occursin(type, "euclid")
            for X = thetaXs density = density + exp.( 30*(bins .- X).^2 ) end
        else
            for X = thetaXs density = density + exp.( 30*(cos.(bins .- X) .- 1) ) end
            θmax = bins[argmax(density)]
            thetaXs[thetaXs .< (θmax-pi)] .+= 2*pi
            thetaXs[thetaXs .> (θmax+pi)] .-= 2*pi
        end
        vals[:, i] = quantile(thetaXs, qs)
        vals[2, i] = mean(thetaXs); vals[1, i] = mean(thetaXs)-2*std(thetaXs); vals[3, i] = mean(thetaXs)+2*std(thetaXs)
    end

    #plot the result
    figure(figsize = figsize)
    if length(plotps) > 0 plot(1:length(plotinds), plotps[plotinds], "b-", markersize = 3, alpha = 0.95) end
    plot(1:length(plotinds), vals[2,:], "g-", markersize = 3, alpha = 0.95)
    fill_between(1:length(plotinds), vals[1,:], vals[3,:], color = "g", alpha = 0.2)
    xticks([], [])
    xlim(1, length(plotinds))
    if occursin("euclid", type)
        ylabel("position")
    else
        ylabel(L"\theta")
        yticks([0; pi; 2*pi], [L"$0$", L"$ \pi $", L"$ 2 \pi $"])
    end
    legend(["head direction", "latent posterior"])
    savefig("figures/"*fname*"thetas"*ext, bbox_inches = "tight")
    close()
    if ext == ".pdf" #write to file
        if length(plotps) > 0 tosave = [1:length(plotinds) plotps[plotinds] vals[2, :] vals[1, :] vals[3, :]]
        else tosave = [1:length(plotinds) vals[2, :] vals[1, :] vals[3, :]] end
        writedlm("figures/"*fname*"thetas.tsv", tosave)
    end
    return vals
end
