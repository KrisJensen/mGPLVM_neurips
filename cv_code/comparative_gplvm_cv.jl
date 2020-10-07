"""
Run crossvalidation calculations on synthetic data
10 repetitions, 50/50 train/test
"""

using JLD, DelimitedFiles, Random
include(joinpath(@__DIR__, "../src/", "fit_gplvm.jl"))
include(joinpath(@__DIR__, "../src/", "fitting_data.jl"))
include(joinpath(@__DIR__, "../src/", "initialize_parameters.jl"))

Random.seed!(20042405)

T2 = true
SO3 = false

if T2
    fit_manifolds = ["2-torus", "2-cube"]
elseif SO3
    fit_manifolds = ["SO3", "3-cube"]
else
    println("No ground truth selected")
end

n1, metric, ntilde = get_params(fit_manifolds[1])

if n1 == 2
    N, T, mind, m, nmax, minH = 150, 200, 30, 100, 1000, 300
else
    N, T, mind, m, nmax, minH = 200, 200, 30, 100, 1000, 300
end
alpha = 0.025 #learning rate
nmin = 1
niters = 10
fname = "cv/"*fit_manifolds[1]

for niter = nmin:niters #for each iteration
    println("generating data")
    n, metric, ntilde = get_params(fit_manifolds[1]);  #generate new data each trial
    xs, ps, Y, alphas, betas, ls, sigs, gammas = gen_data(N, T, n, metric = metric, uncertainty = "high") #draw data from true manifold
    true_params = [xs, ps, Y, alphas, betas, ls, sigs, gammas]

    Ts = randperm(T)
    fitTs = 1:2:T
    testTs = 2:2:T
    #split neurons into two halves for cross validation
    Xs1 = 1:2:N
    Xs2 = 2:2:N

    for fit_manifold = fit_manifolds
        if fit_manifold == "SO3" kmax = 5 else kmax = 3 end
        n, metric, ntilde = get_params(fit_manifold);
        println("\n\n cross-validating ", fit_manifold, " to ", fname, " ", niter)

        #fit half of the conditions using all of the neurons
        fitY = Y[:, fitTs]
        μs, Σs, alphas0, ls0, sigs0, us, dfunc = initialize_parameters(fit_manifold, fitY, n, T, mind) #initialize things
        ls0 = [mean(ls0)]; sigs0 = [mean(sigs0)]; alphas0 = [mean(alphas0)]; us = [us[1]]
        Lfit, θfit = fit_gplvm(fitY, μs, Σs, alphas0, ls0, sigs0, us, n, dfunc, m=m, nmax = nmax, alpha=alpha, kmax = kmax, minH = minH, comb=true)

        μs, Σs, _, _, _, _, _ = initialize_parameters(fit_manifold, Y, n, T, mind) #initialize things
        alphas0, ls0, sigs0, us = θfit[3:end] #fix the kernels
        μs[fitTs] = θfit[1][:] #fix what we've already learned
        Σs[fitTs] = θfit[2][:] #fix what we've already learned

        #fit missing variational distributions using only Xs1
        L1, θcv1 = fit_gplvm(Y[Xs1, :], μs, Σs, alphas0, ls0, sigs0, us, n, dfunc,
                            m=m, nmax = Int(0.75*nmax), alpha=alpha, kmax = kmax, minH = Int(0.7*minH), fitN = false, fitTs = testTs, comb=true) #only fit new Ts

        μs, Σs, _, _, _, _, _ = initialize_parameters(fit_manifold, Y, n, T, mind) #initialize things
        μs[fitTs] = θfit[1][:] #fix what we've already learned
        Σs[fitTs] = θfit[2][:] #fix what we've already learned
        #Fit missing variational distributions using only Xs2
        L2, θcv2 = fit_gplvm(Y[Xs2, :], μs, Σs, alphas0, ls0, sigs0, us, n, dfunc,
                            m=m, nmax = Int(0.75*nmax), alpha=alpha, kmax = kmax, minH = Int(0.7*minH), fitN = false, fitTs = testTs, comb=true) #only fit new Ts

        #store result
        save("results/"*fname*"_"*fit_manifold*string(niter)*".jld", "fit", θfit, "cv1", θcv1, "cv2", θcv2, "true", true_params, "Lfit", Lfit,
            "Xs1", Xs1, "Xs2", Xs2, "fitTs", fitTs, "testTs", testTs)
    end
end
