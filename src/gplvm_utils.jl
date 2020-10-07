"""
assorted functions for computing the ELBO
"""

using LinearAlgebra, Distributions, Zygote
include("utils.jl")
include("distance_functions_gp.jl")
include("entropy_functions.jl")
include("expmaps.jl")
include("translation_functions.jl")

function calc_H(Xs, μ, Σ, ChS, dfunc, n; kmax = 3)
    #estimate the entropy
    if dfunc == dtheta_gp
        if n == 3
            H_est = entropy_s3(Xs, Σ, ChS, kmax=kmax)
            maxval = log(Ssphere(3, r = 1)) #upper bound is the log volume of S(3)
            return min(H_est, maxval)
        else
            println("not a Lie group, sorry"); return
        end
    elseif dfunc == dtorus_gp
        H_est = entropy_torus(Xs, Σ, ChS, n, kmax=kmax)
        maxval = n*log(2*pi)#upper bound is the log volume of T(n)
        return min(H_est, maxval)
    elseif dfunc == dso3_gp
        H_est = entropy_so3(Xs, Σ, ChS, kmax=kmax)
        maxval = log(Ssphere(3, r = 1)/2) #upper bound is the log volume of SO(3)
        return min(H_est, maxval)
    elseif dfunc == deuclid_gp
        return entropy_euclid(Σ, ChS, n) #analytical
    else
        error("Distance function ", dfunc, " not recognized. Exiting.")
    end
end


function logPX(dfunc, n, X)
    #return the prior logP(X) which is uniform on spheres/tori and Gaussian in Euclidean space
    T = size(X)[2]
    if dfunc == dtheta_gp
        return T*log(1/Ssphere(n)) #here n is the dimensionality of the group (e.g. n=3 for the 3-sphere)
    elseif dfunc == dtorus_gp
        #return T*log(1/(2*pi)^n) #inverse area of the torus
        return -T*n*log(2*pi)
    elseif dfunc == deuclid_gp
        #P(X) ~ N(0, I), det(I) = 1
        P = -T*n/2*log(2*pi) #T lots of 1/sqrt((2*pi)^n |I|)
        P = P - 0.5 * sum(X.^2) #sum{ log(exp(- 1/2 x' I x)) } = -0.5 sum{x_it^2}
        return P
    elseif dfunc == dso3_gp
        return T*log(1/(Ssphere(3)/2)) #inverse area of 3-hemisphere
    end
end

function logP(Y, X, alphas, ls, sigs, dfunc, us; comb = false)
    #return logP(X, Y) = logP(Y|X) + logP(X)
    n, T = size(X)
    if comb
        logPY_X = calc_sparse_GP_L_comb(Y, X, alphas, ls, sigs, us, dfunc) #tie GP hyperparams across neurons
    else
        logPY_X = calc_sparse_GP_L(Y, X, alphas, ls, sigs, us, dfunc)
    end
    logP_X = logPX(dfunc, n, X) #one for each datapoint
    return logPY_X+logP_X
end

cat3(x1, x2) = cat(x1, x2, dims = 3) #syntactic sugar to use in reduce(cat, ...) statements

function fixedcat(vals, n, T, m)
    #helper function to concatenate some data in zygote
    buf = Zygote.Buffer(zeros(m, n, T), m, n, T)
    for i = 1:T
        buf[:, :, i] = vals[i][:, :]
    end
    return copy(buf)
end

function expmap(Xs, dfunc)
    #calls an appropriate exponential map function
    if dfunc == dtorus_gp
        gs = expmap_torus(Xs)
    elseif dfunc == deuclid_gp
        gs = expmap_euclid(Xs)
    elseif dfunc == dtheta_gp
        gs = expmap_s3(Xs) #only implemented for glome since S1=T1 and the others are not Lie groups. See separate OCaml implementation for other spheres
    elseif dfunc == dso3_gp
        gs = expmap_so3(Xs)
    else
        exit("distance function not recognizeed sorry")
    end
    return gs
end

function translate_group(gs, μs, dfunc)
    #calls an appropriate function to apply group element g^mu
    if dfunc == dtorus_gp
        gs = translate_torus(gs, μs)
    elseif dfunc == deuclid_gp
        gs = translate_euclid(gs, μs)
    elseif dfunc == dtheta_gp
        gs = translate_s3(gs, μs)
    elseif dfunc == dso3_gp
        gs = translate_so3(gs, μs)
    else
        exit("distance function not recognizeed sorry")
    end
end

function calc_ELBO(Y, Xtildes, μs, Σs, alphas, ls, sigs, dfunc, entropy, us, kmax; comb = false)
    #return ELBO = E_Q[logP(X, Y)] + H(Q). The energy term is computed via monte carlo sampling
    T = length(Xtildes)
    m, n = size(Xtildes[1])
    ChS = [cholesky(Σs[j] + 1e-8*I) for j = 1:T] #cholesky the covariance matrices
    Xs = [Xtildes[j]*ChS[j].U for j = 1:T] #mu = 0; each of these is mxn

    if entropy #compute entropies as in Falorsi et al.
        H = sum([calc_H(Xs[j], μs[j], Σs[j], ChS[j], dfunc, n, kmax = kmax) for j = 1:T]) #entropy H=E_Q[logQ] ###calculatee entropy properly
    else
        H = 0 #only consider datafit term
    end
    
    gs = [expmap(Xs[j], dfunc) for j = 1:T] #project onto group using the exponential map
    gs = [translate_group(gs[j], μs[j], dfunc) for j = 1:T] #apply group element to shift the distribution
    gs = fixedcat(gs, length(μs[1]), T, m) #m x n x T

    E = sum([logP(Y, gs[i, :, :], alphas, ls, sigs, dfunc, us, comb = comb) for i = 1:m]) #compute the log likelihood
    L = H + E/m #ELBO
    return L/T #normalize by number of timepoints
end
