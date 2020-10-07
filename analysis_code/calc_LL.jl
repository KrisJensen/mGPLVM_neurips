"""
code for computing importance weighted log likelihoods
(Burda 2015)
"""

using LinearAlgebra, StatsFuns
include(joinpath(@__DIR__, "../src/", "gplvm_utils.jl"))
include("individual_entropy_functions.jl")

function calc_LL(θ, Y, n, dfunc; m = 1000, comb = false, kmax = 3)
    N, T = size(Y)
    μs, alphas, ls, sigs, us = θ[1], θ[3], θ[4], θ[5], θ[6] #fitted hyperprams
    Σs = [Σ for Σ  = θ[2]]

    Xtildes = [randn(m, n) for i =1:T]#randn(n, T, m)
    ChS = [cholesky(Σs[j] + 1e-8*I) for j = 1:T] #cholesky the covariance matrices
    Xs = [Xtildes[j]*ChS[j].U for j = 1:T] # each of these is mxn

    #compute logQs
    logQs = [i_calc_H(Xs[j], μs[j], Σs[j], ChS[j], dfunc, n, kmax = kmax) for j = 1:T]
    logQs = reduce(hcat, logQs)
    logQs = sum(logQs, dims = 2)[:] #sum over conditions

    gs = [expmap(Xs[j], dfunc) for j = 1:T] #project onto group using the exponential map
    gs = [translate_group(gs[j], μs[j], dfunc) for j = 1:T] #apply group element to shift the distribution
    gs = fixedcat(gs, length(θ[1][1]), T, m) #m x n x T

    #compute log p(Y, {g})
    logPs = [logP(Y, gs[i, :, :], alphas, ls, sigs, dfunc, us, comb = comb) for i = 1:m]
    logvals = [logPs[i] + logQs[i] for i = 1:m]
    LL = logsumexp(logvals)-log(m) #expectation

    return LL/T
end
