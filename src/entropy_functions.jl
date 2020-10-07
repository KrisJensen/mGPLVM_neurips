"""
functions for computing the entropies of distributions
on various manifolds
"""

using LinearAlgebra

function entropy_torus(Xs, Σ, ChS, n; kmax=3)
    #Xs is an mxn array
    #sigma is a covariance matrix
    #ChS is its cholesky decomposition
    Xs = (Xs .+ 2*pi) .% (2*pi) #center the input
    ks = -kmax:1:kmax
    logS = 2*sum(log.(diag(ChS.L))) #log |Σ|
    if n == 1 #can do this using the pointwise pdf
        Xs = Xs .+ (ks .* 2*pi)'
        qhat = sum(exp.(-Xs.^2/(2*Σ[1])), dims = 2) #qhat(g(x))
    else #loop over combinations of ks
        qhat = zeros(size(Xs)[1])
        if n == 2
            for k1 = ks for k2 = ks
                newXs = Xs .+ ([k1; k2]*2*pi)' #move around
                qhat = qhat + exp.( -0.5 * sum(newXs' .* (ChS.U \ (ChS.L \ newXs')), dims = 1) )[:] #add to the cumulative probability
            end end
        elseif n == 3
            for k1 = ks for k2 = ks for k3 = ks
                newXs = Xs .+ ([k1; k2; k3]*2*pi)' #move around
                qhat = qhat + exp.( -0.5 * sum(newXs' .* (ChS.U \ (ChS.L \ newXs')), dims = 1) )[:] #add to the cumulative probability
            end end end
        elseif n == 4
            for k1 = ks for k2 = ks for k3 = ks for k4 = ks
                newXs = Xs .+ ([k1; k2; k3; k4]*2*pi)' #move around
                qhat = qhat + exp.( -0.5 * sum(newXs' .* (ChS.U \ (ChS.L \ newXs')), dims = 1) )[:] #add to the cumulative probability
            end end end end
        else exit("n = "*string(n)*" not implemented") end
    end
    H_est = -mean(log.(qhat)) + 0.5*logS + n/2*log(2*pi)
    return H_est
end

function entropy_euclid(Σ, ChS, n)
    #return the analytical entropy
    logS = 2*sum(log.(diag(ChS.L)))
    return n/2*log(2*pi*exp(1)) + 0.5*logS
end

function calcJ(ϕ)
    ϕ = max(ϕ, 1e-4)
    J = ϕ^2/(2 - 2*cos(ϕ))
    J = min(1e5, J) #things diverge when ϕ>0 and cos(ϕ)->1. Measure zero in theory, numerical issues in practice. These points have infinite negative entropy so not our solution anyways
    return J
end

function entropy_so3(Xs, Σ, ChS; kmax=3)
    #compute entropy of SO(3) using eq in paper
    logS = 2*sum(log.(diag(ChS.L)))
    ks = -kmax:1:kmax
    Nk = length(ks)
    m = size(Xs)[1]
    θs = sqrt.(sum(Xs.^2, dims = 2)) #norms
    us = Xs ./ θs
    qhat = zeros(m)
    for k = ks #finite terms of infinite sum
        newXs = Xs .+ (k*pi*us) #move around
        newp = exp.( -0.5 * sum(newXs' .* (ChS.U \ (ChS.L \ newXs')), dims = 1) )[:] #pdfs
        ϕs = 2. * sqrt.(sum(newXs.^2, dims = 2))[:]
        Js = calcJ.(ϕs)
        qhat = qhat + Js.*newp  #add to the cumulative probability
    end
    H_est = -mean(log.(qhat)) + 0.5*logS + 3/2*log(2*pi) #add normalization
    return H_est
end

function entropy_s3(Xs, Σ, ChS; kmax=3)
    #S^3 entropy is same as SO(3) except the sum runs over 2*pi*k
    logS = 2*sum(log.(diag(ChS.L)))
    ks = -kmax:1:kmax
    Nk = length(ks)
    m = size(Xs)[1]
    θs = sqrt.(sum(Xs.^2, dims = 2)) #noorms
    us = Xs ./ θs
    qhat = zeros(m)
    for k = ks
        newXs = Xs .+ (2*k*pi*us) #q != -q on s3
        newp = exp.( -0.5 * sum(newXs' .* (ChS.U \ (ChS.L \ newXs')), dims = 1) )[:] #pdfs
        ϕs = 2. * sqrt.(sum(newXs.^2, dims = 2))[:]
        Js = calcJ.(ϕs)
        qhat = qhat + Js.*newp  #add to the cumulative probability
    end
    H_est = -mean(log.(qhat)) + 0.5*logS + 3/2*log(2*pi)
    return H_est
end
