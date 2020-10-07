"""
functions for generating synthetic data on various manifolds.
synthetic data takes the form
y_ij = a_i exp( d_geo^2(g_i^pref, g_j) / (2*b_i^2) ) + c_i
uses geodesic distance functions from distance_functions.jl
"""

using LinearAlgebra, Statistics, MultivariateStats
include("distance_functions.jl")
include("distance_functions_gp.jl")
include("utils.jl")

function sim_xs_ps(N, T, n; step = 0.2, metric = "theta", cont = false)
    #generates latents "ps" and preferred directions "xs" for synthetic data
    #if cont, the latents form a continuous random walk for ease of visualization
    if metric == "theta"
        ps = randn(n+1, T)
        ps = ps ./ sqrt.(sum(ps.^2, dims = 1)) #true latent states
        xs = randn(N, n+1) #true receptive fields
        xs = xs ./ sqrt.(sum(xs.^2, dims = 2)) #randomly sample unit vectors
        if ~cont return xs, ps end #just return random points
    elseif metric == "euclid"
        ps = rand(n, T)*3
        xs = rand(N, n)*3 #randomly sample the unit cube
        if ~cont return xs, ps end
    elseif metric == "torus"
        ps = rand(n, T)*2*pi #direct product of n circles
        xs = rand(N, n)*2*pi #direct product of n circles
        if ~cont return xs, ps end
    elseif metric == "SO3"
        n = 4
        ps = randn(n, T)
        ps = ps ./ sqrt.(sum(ps.^2, dims = 1)) .* sign.(ps[1, :])'
        xs = randn(N, n)
        xs = xs ./ sqrt.(sum(xs.^2, dims = 2)) .* sign.(xs[:, 1])
        return xs, ps
    else
        println("metric not recognized"); return
    end

    #random wallk on the sphere
    if metric == "theta"
        for t = 2:T
            s0 = ps[:, t-1]
            s1 = randn(n)
            s1 = s1 - (s0' * s1)*s0 #tangent vector
            s1 = s1/norm(s1) * rand()*step #randoms step size with norm between -0.1 and 0.1
            ps[:, t] = (s0+s1)/norm(s0+s1)
        end #random walk
    else
        for t = 2:T
            s0 = ps[:, t-1]
            s1 = randn(n)
            s1 = s1/norm(s1) * rand()*step*0.5 #randoms step size with norm between -0.1 and 0.1
            if metric == "euclid"
                s1[abs.(s0) .> 1.5] .=  (-sign.(s0) .* abs.(s1))[abs.(s0) .> 1.5] #bound the random walk
            end
            news = s0+s1
            if metric == "torus"
                if n == 1 xs = reshape(Array(range(2*pi/N, 2*pi, length = N)), N, n) end
                news[news .< 0] .+= 2*pi
                news[news .> 2*pi] .-= 2*pi
            end
            ps[:, t] = news
        end #random walk
        if metric == "euclid" xs = rand(N, n)*3 .- 1.5 end
    end
    return xs, ps
end


function gen_data(N, T, n; metric = "theta", cont = false, f = "default", uncertainty = "high", threefold = false)
    #generate data on a manifold
    #uncertainty can be high (used for crossvalidation) or low (used for example plots)
    xs, ps = sim_xs_ps(N, T, n, step = 1, metric = metric, cont = cont) #receptive fields and latents
    gammas = ones(T) #.+ 0.2*sin.( 6*pi*(1:T)./T ) #global gains
    if uncertainty == "high"
        alphas = 1 .+ randn(N)*0.2 #true gains
        betas = rand(N)*0.2
        sigs = 0.20 .+ randn(N)*0.020 #true observation noise
    else
        alphas = 1 .+ randn(N)*0.05 #true gains
        betas = rand(N)*0.05
        sigs = 0.1 .+ randn(N)*0.001 #true observation noise
    end

    if metric == "theta"
        if f == "default"
            if uncertainty == "low" f = 0.14 else f = 0.07 end
        end
        l = ( 1/Vball(n) * Ssphere(n) * f )^(1/n)
        #fraction of points in vicinity d is Vball(n, r=l)/Ssphere(n)
        dfunc = dtheta
    elseif metric == "euclid"
        if f == "default"
            f = 0.13
        end
        l = 3 * ( 1/Vball(n) * f )^(1/n)
        #fraction of points in vicinity d is Vball(n)/x^n
        dfunc = deuclid
    elseif metric == "torus"
        if f == "default"
            f = 0.15
            if n == 1 f = 0.14 end
            if n == 2 f = 0.12 end
            if uncertainty == "low" f = 0.14 end
        end
        l = 2*pi * ( 1/Vball(n) * f )^(1/n)
        dfunc = dtorus
    elseif metric == "SO3"
        if f == "default" f = 1.0 end
        if uncertainty == "low" f = 1.3 end
        l = f
        dfunc = dso3
    end
    if uncertainty == "high"
        ls = l .+ rand(N)*0.15*l
        noised = true #determines whether we add noise to the computed geodesic distances
    else
        ls = l .+ rand(N)*0.05*l
        noised = false
    end
    fs = calcg(xs, ps, ls, alphas, betas, gammas, dfunc = dfunc, noised = noised)
    ys = fs + randn(N, T).*sigs
    return xs, ps, ys, alphas, betas, ls, sigs, gammas
end
