"""
Run crossvalidation calculations on Drosophila data
10 repetitions, 50/50 train/test
"""

using JLD, DelimitedFiles, Random
include(joinpath(@__DIR__, "../src/", "fit_gplvm.jl"))
include(joinpath(@__DIR__, "../src/", "fitting_data.jl"))
include(joinpath(@__DIR__, "../src/", "initialize_parameters.jl"))

type = 1 #spread this out over two calculations to speed things up a bit (this was slightly last minute)
if type == 1
    Random.seed!(20042405)
    iternums = 1:7
elseif type == 2
    Random.seed!(13400206)
    iternums = 8:10
end

fname = "fly1_trial1_light"

fit_manifolds = ["1-torus", "1-cube"] #manifolds to fit
mind, m, nmax, minH = 15, 100, 360, 100
kmax = 3
niters = 10

##load data
Y = readdlm("fly_data/"*fname*"_Y.tsv")
xs = readdlm("fly_data/"*fname*"_x.tsv")'
true_params = [xs, Y] #note that the xs are actually what we call ps (true latents; in this case head direction)
N, T = size(Y)
alpha = 0.4e-1 #learning rate


for niter = iternums
    Ts = randperm(T) #random 50/50 train/test split
    fitTs = Ts[1:Int(length(Ts)/2)]
    testTs = Ts[Int(length(Ts)/2+1):end]
    #split neurons into two halves for cross validation
    Xs1 = 1:2:N
    Xs2 = 2:2:N
    for fit_manifold = fit_manifolds
        n, metric, ntilde = get_params(fit_manifold);
        println("\n\n cross-validating ", fit_manifold, " to ", fname, " ", niter)

        #fit half of the conditions using all of the neurons
        fitY = Y[:, fitTs]
        μs, Σs, alphas0, ls0, sigs0, us, dfunc = initialize_parameters(fit_manifold, fitY, n, T, mind) #initialize things
        Lfit, θfit = fit_gplvm(fitY, μs, Σs, alphas0, ls0, sigs0, us, n, dfunc, m=m, nmax = nmax, alpha=alpha,
                                        kmax = kmax, minH = minH, thresh = 0.005)


        μs, Σs, _, _, _, _, _ = initialize_parameters(fit_manifold, Y, n, T, mind) #initialize things
        alphas0, ls0, sigs0, us = θfit[3:end] #fix the kernels
        μs[fitTs] = θfit[1][:] #fix what we've already learned
        Σs[fitTs] = θfit[2][:] #fix what we've already learned

        #fit missing variational distributions using only Xs1
        L1, θcv1 = fit_gplvm(Y[Xs1, :], μs, Σs, alphas0[Xs1], ls0[Xs1], sigs0[Xs1], us[Xs1], n, dfunc,
                            m=m, nmax = Int(0.75*nmax), alpha=alpha, kmax = kmax, minH = minH, fitN = false,
                            fitTs = testTs, thresh = 0.005) #only fit new Ts

        μs, Σs, _, _, _, _, _ = initialize_parameters(fit_manifold, Y, n, T, mind) #initialize things
        μs[fitTs] = θfit[1][:] #fix what we've already learned
        Σs[fitTs] = θfit[2][:] #fix what we've already learned
        #Fit missing variational distributions using only Xs2
        L2, θcv2 = fit_gplvm(Y[Xs2, :], μs, Σs, alphas0[Xs2], ls0[Xs2], sigs0[Xs2], us[Xs2], n, dfunc,
                            m=m, nmax = Int(0.75*nmax), alpha=alpha, kmax = kmax, minH = minH, fitN = false,
                            fitTs = testTs, thresh = 0.005) #only fit new Ts

        #store result
        save("results/fly_cv/"*fname*"_"*fit_manifold*string(niter)*".jld", "fit", θfit, "cv1", θcv1, "cv2", θcv2, "true", true_params, "Lfit", Lfit,
            "Xs1", Xs1, "Xs2", Xs2, "fitTs", fitTs, "testTs", testTs)
    end
end
