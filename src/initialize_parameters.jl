"""
function for initializing models appropiately.
q(g) are initialized at the identity with ~large variance.
inducing points are initialized accoording to the prior.
"""

using LinearAlgebra

function initialize_parameters(fit_manifold, Y, n, T, mind; comb = false)
    #initialization function for fitting gplvm
    #"fit_manifold" is the manifold to fit

    N, T = size(Y)
    μs = [zeros(n) for i = 1:T] #zero initial mean

    if occursin("torus", fit_manifold)
        dfunc = dtorus_gp
        scale = pi/2*sqrt(n) #standard deviation of initial approximating distribution
        us = [rand(n, mind)*2*pi for i = 1:N] #initialize the inducing points uniformly; separate points for each neuron
    elseif occursin("sphere", fit_manifold)
        if n == 3
            scale = 0.50 #don't want saturated entropy yet
            dfunc = dtheta_gp
            us = [randn(4, mind) for i = 1:N] #initialize the inducing points uniformly
            us = [u ./ sqrt.(sum(u.^2, dims = 1)) for u = us] #project out double coverage
            μs = [[1.; 0.; 0.; 0.] for i = 1:T] #initial means are the identity operation
        else
            println("not a group, sorry")
            return
        end
    elseif occursin("SO3", fit_manifold)
        #scale = 0.30 #don't want saturated entropy yet
        scale = 0.40 #don't want saturated entropy yet
        dfunc = dso3_gp
        us = [randn(4, mind) for i = 1:N] #initialize the inducing points uniformly
        us = [sign.(u[1, :])' .* u ./ sqrt.(sum(u.^2, dims = 1)) for u = us] #project out double coverage
        μs = [[1.; 0.; 0.; 0.] for i = 1:T] #initial means are the identity operation
    elseif occursin("cube", fit_manifold)
        scale = 1.0
        dfunc = deuclid_gp
        us = [randn(n, mind) for i = 1:N] #distribute inducing points according to the prior N(0, I)
    else
        println("manifold ", fit_manifold, " not recognized, sorry"); return
    end

    Σs = [Symmetric(diagm(ones(n)))*scale for i = 1:T] #initial variational distributions are broad
    alphas0 = mean(Y, dims = 2)[:]; sigs0 = std(Y, dims = 2)[:]; ls0 = ones(N) #initialize some parameters
    alphas0 = max.(alphas0, 0.05)
    if comb ls0 = [mean(ls0)]; sigs0 = [mean(sigs0)]; alphas0 = [mean(alphas0)]; us = [us[1]] end

    return μs, Σs, alphas0, ls0, sigs0, us, dfunc

end
