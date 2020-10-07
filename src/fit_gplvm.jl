"""
main function for fitting a gplvm model
requires initial guesses for all parameters as well as a learning rate,
number of samples for monte carlo estimates and convergence criteria
"""

using Zygote, LinearAlgebra
include("distance_functions_gp.jl")
include("fitting_data.jl")
include("gplvm_utils.jl")


function fit_gplvm(Y, μs, Σs, alphas, ls, sigs, us, n, dfunc; m=100, nmax = 300, alpha = 0.5e-1, kmax = 3, comb = false,
                    minH = 30, fitN = true, fitTs = "default", thresh = 0.01)
    #Y: data
    #μs & Σs: initial means and covariance matrices for variational distributions
    #initial values of alphas, ls and sigs
    #initial inducing points us
    #dfunc specifies the manifold-specific GP kernel (deuclid_gp, dtorus_gp, dso3_gp, dtheta_gp)
    #m is number of monte carlo samples for each gradient step
    #nmax is the max number of iterations, thresh is the convergence threshold
    #alpha is the ADAM learning rate
    #kmax is the number of terms to include to compute q_tilde
    #comb determines whether GP parameters are tied across neurons
    #minH is the length of the burn-in period
    #if not fitN, neuron-specific parameters are fixed (used for cv)
    #if fitTs != "default", we only update variational distributions for these time points

    Ls = [cholesky(Σ).L for Σ = Σs] #cholesky all these cool cats and kittens

    alphas = sqrt.(alphas) #let calc_ELBO square the alphas and then square them again at the end

    N, T = size(Y)
    t0 = time() #duration of optimization
    if fitTs == "default" fitTs = 1:T end #timesteps to optimize
    println("fitting time points ", fitTs)

    θs = [μs, Ls, alphas, ls, sigs, us] #parameters to be optimized
    beta1, beta2, eps = 0.9, 0.999, 1e-8 #learning hyperparameters for ADAM
    ns = [T;T;0;0;0;length(us)]
    ms, vs = [[(if (i in 3:5) zeros(size(θs[i])) else [zeros(size(θs[i][j])) for j = 1:ns[i]] end) for i = 1:6] for k = 1:2] #first and second moment estimates for ADAM
    ms[2], vs[2] = [ [LowerTriangular(zeros(size(θs[2][j]))) for j = 1:T] for k = 1:2] #crappy code sorry whoever reads this

    Loss, Hs, Es = zeros(10), zeros(10), zeros(10) #keep track of some progress

    #square alphas here to ensure positive
    fbase(Xtildes, μs, Ls, alphas, ls, sigs, entropy, us) = calc_ELBO(Y, Xtildes, μs,
                                                                    [(L*L' + 1e-4*I) for L = Ls],
                                                                    alphas.^2, ls, sigs, dfunc, entropy, us, kmax, comb = comb) #function to be optimized

    epoch = 0
    minalph = Int(round(minH*1.0))
    while (epoch <= (minH+10)) || ( (epoch < nmax) & (std(Loss)>thresh) ) #finish at nmax or when std(Loss) <= thresh
            Xtildes = [randn(m, n) for i =1:T]#randn(n, T, m)
        epoch += 1

        if epoch < minH #only fit data to begin with
            alphaeff = alpha
            entropy = false
            H = 0
        else #add entropy term later
            alphaeff = alpha
            entropy = true
            ChS = [cholesky(θs[2][i]*θs[2][i]' + 1e-4*I) for i = 1:T] #cholesky the covariance matrices
            Xs = [Xtildes[i]*ChS[i].U for i = 1:T] #mu = 0
            H = sum([calc_H(Xs[j], θs[1][j], θs[2][j]*θs[2][j]'+1e-4*I, ChS[j], dfunc, n, kmax = kmax) for j = 1:T])/T
        end

        f(μs, Ls, alphas, ls, sigs, us) = fbase(Xtildes, μs, Ls, alphas, ls, sigs, entropy, us)
        L = f(θs[1], θs[2], θs[3], θs[4], θs[5], θs[6])
        grads = gradient(f, θs[1], θs[2], θs[3], θs[4], θs[5], θs[6])

        Loss[epoch % 10 + 1] = L #store some numbers
        Hs[epoch % 10 + 1] = H
        Es[epoch % 10 + 1] = Loss[epoch % 10 + 1]-Hs[epoch % 10 + 1]
        L = sum(Loss) / min(epoch, 10); E = sum(Es) / min(epoch, 10)
        H = sum(Hs) / min((epoch-minH+1), 10)

        alpha_hat = alphaeff*sqrt(1-beta2^epoch)/(1-beta1^epoch) #bias correction of step size
        eps_hat = sqrt(1-beta2^epoch)*eps

        for ind = fitTs #variational distributions

            #means
            g = grads[1][ind][:]
            ms[1][ind] = beta1*ms[1][ind] + (1-beta1)*g #update first moment
            vs[1][ind] = beta2*vs[1][ind] + (1-beta2)*(g.^2) #update second moment
            θs[1][ind] = θs[1][ind] + alpha_hat * ms[1][ind] ./ (sqrt.(vs[1][ind]) .+ eps_hat)
            #covs
            if epoch >= minH
                g = LowerTriangular(grads[2][ind]) #symmetrize
                ms[2][ind] = beta1*ms[2][ind] + (1-beta1)*g #update first moment
                vs[2][ind] = beta2*vs[2][ind] + LowerTriangular((1-beta2)*(g.^2)) #update second moment
                θs[2][ind] = θs[2][ind] + LowerTriangular(alpha_hat * ms[2][ind] ./ (sqrt.(vs[2][ind]) .+ eps_hat))
            end
        end

        if dfunc == dso3_gp #normalize 'means' of variational distribution
            for ind = fitTs θs[1][ind] = θs[1][ind] / norm(θs[1][ind]) end
        end

        if fitN #if we fit neuron data
            for ind = 1:length(θs[6]) #inducing points for each neuron
                g = grads[6][ind]
                ms[6][ind] = beta1*ms[6][ind] + (1-beta1)*g #update first moment
                vs[6][ind] = beta2*vs[6][ind] + (1-beta2)*(g.^2) #update second moment
                θs[6][ind] = θs[6][ind] + alpha_hat * ms[6][ind] ./ (sqrt.(vs[6][ind]) .+ eps_hat)
            end

            for i = 4:5 #update neuron parameters
                ms[i] = beta1*ms[i] + (1-beta1)*grads[i] #update first moment
                vs[i] = beta2*vs[i] + (1-beta2)*(grads[i].^2) #update second moment
                θs[i] = θs[i] + alpha_hat * ms[i] ./ (sqrt.(vs[i]) .+ eps_hat)
            end
            if epoch >= minalph
                ms[3] = beta1*ms[3] + (1-beta1)*grads[3] #update first moment
                vs[3] = beta2*vs[3] + (1-beta2)*(grads[3].^2) #update second moment
                delta = alpha_hat * ms[3] ./ (sqrt.(vs[3]) .+ eps_hat)
                θs[3] = θs[3] + sign.(delta) .* min.(abs.(θs[3])*0.1, abs.(delta))
            end

            if dfunc in [dtheta_gp; dso3_gp] #normalize inducing points
                for ind = 1:length(θs[6]) θs[6][ind] = θs[6][ind] ./ sqrt.(sum(θs[6][ind].^2, dims = 1)) end
            end
        end

        if dfunc == dso3_gp
            printmu = round(mean( [sqrt(μ[2:end]'*μ[2:end]) for μ = θs[1]] ), digits = 2)
        else
            printmu = round(mean( [sqrt(μ'*μ) for μ = θs[1]] ), digits = 2)
        end
        println(epoch, " L=", round(L, digits=2), " E=", round(E, digits=2), " H=", round(H, digits=2),
                " t=", round(time()-t0),
                " mu=", printmu,
                " cov=", round(median(reduce(vcat, [diag(L*L') for L = θs[2]])), digits = 4),
                " alp=", round(mean(θs[3].^2), digits=3), " noise=", round(mean(θs[5]), digits=3),
                " l=", round(mean(θs[4]), digits=3),
                " lrate=", round(alphaeff, digits=5))
    end
    Xtildes = [randn(10*m, n) for i =1:T] #big sample to compute final L
    L = fbase(Xtildes, θs[1], θs[2], θs[3], θs[4], θs[5], true, θs[6])
    println("final L:", L)
    θs[3] = θs[3].^2 #use squared alpha in inference functions for prior compatibility
    θs[2] = [Symmetric(θs[2][j]*θs[2][j]') for j = 1:length(θs[2])]
    return L, θs #optimized parameters
end
