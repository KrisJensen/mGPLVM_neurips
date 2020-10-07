"""
Code for computing logq(g) for each g individually instead
of averaged over monte carlo samples
Needed for calcLL but mostly copied from src/entropy_functions.jl
Apologies for the bad conding practice.
"""

using LinearAlgebra

function i_calc_H(Xs, μ, Σ, ChS, dfunc, n; kmax = 3)
    if dfunc == dtheta_gp #project variational distribution onto tangent plane
        if n == 3
            H_est = i_entropy_s3(Xs, Σ, ChS, kmax=kmax)
            return H_est
        else
            println("not a Lie group, sorry"); return
        end
    elseif dfunc == dtorus_gp
        H_est = i_entropy_torus(Xs, Σ, ChS, n, kmax=kmax)
        return H_est
    elseif dfunc == dso3_gp
        H_est = i_entropy_so3(Xs, Σ, ChS, kmax=kmax)
        return H_est
    elseif dfunc == deuclid_gp
        return i_entropy_euclid(Xs, Σ, ChS, n)
    else
        error("Distance function ", dfunc, " not recognized. Exiting.")
    end
end

function i_entropy_torus(Xs, Σ, ChS, n; kmax=3)
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
    H_est = -log.(qhat) .+ 0.5*logS .+ n/2*log(2*pi)
    return H_est
end

function i_entropy_euclid(Xs, Σ, ChS, n)
    #return the analytical entropy
    #ChS is its cholesky decomposition
    logS = 2*sum(log.(diag(ChS.L))) #log |Σ|
    logqhat = -0.5 * sum(Xs' .* (ChS.U \ (ChS.L \ Xs')), dims = 1)[:]
    H_est = -logqhat .+ 0.5*logS .+ n/2*log(2*pi)

    return H_est

    logS = 2*sum(log.(diag(ChS.L)))
    return n/2*log(2*pi*exp(1)) + 0.5*logS
end


# function calcJ(ϕ)
#     if ϕ < 1e-4 return 1 end
#     return ϕ^2/(2 - 2*cos(ϕ))
# end

function i_calcJ(ϕ)
    ϕ = max(ϕ, 1e-4)
    J = ϕ^2/(2 - 2*cos(ϕ))
    J = min(1e5, J) #things diverge when ϕ>0 and cos(ϕ)->1. Measure zero in theory, numerical issues in practice. These points have infinite negative entropy so not our solution anyways
    return J
end

function i_entropy_so3(Xs, Σ, ChS; kmax=3)
    logS = 2*sum(log.(diag(ChS.L)))
    ks = -kmax:1:kmax
    Nk = length(ks)
    m = size(Xs)[1]
    θs = sqrt.(sum(Xs.^2, dims = 2)) #noorms
    us = Xs ./ θs
    qhat = zeros(m)
    for k = ks
        newXs = Xs .+ (k*pi*us) #move around
        newp = exp.( -0.5 * sum(newXs' .* (ChS.U \ (ChS.L \ newXs')), dims = 1) )[:] #pdfs
        ϕs = 2. * sqrt.(sum(newXs.^2, dims = 2))[:]
        Js = i_calcJ.(ϕs)
        qhat = qhat + Js.*newp  #add to the cumulative probability
    end
    H_est = -log.(qhat) .+ 0.5*logS .+ 3/2*log(2*pi)
    return H_est
end

function i_entropy_s3(Xs, Σ, ChS; kmax=3)
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
        Js = i_calcJ.(ϕs)
        qhat = qhat + Js.*newp  #add to the cumulative probability
    end
    H_est = -log.(qhat) .+ 0.5*logS .+ 3/2*log(2*pi)
    return H_est
end
