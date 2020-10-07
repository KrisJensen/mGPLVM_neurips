"""
code for analyzing the results of crossvalidation calculations
computes both MSE and test LL
"""

using PyPlot, JLD, LinearAlgebra
rc("font", family="sans-serif", size = 12)

include(joinpath(@__DIR__, "../src/", "gplvm_utils.jl"))
include(joinpath(@__DIR__, "../src/", "initialize_parameters.jl"))
include(joinpath(@__DIR__, "../analysis_code/", "calc_LL.jl"))
include(joinpath(@__DIR__, "../analysis_code/", "gplvm_inference.jl"))

#fly or synthetic data
T2 = true
SO3 = false
threefold = false
flycalc = false

addstr = "" #labelling
comb = true #tie parameters

if flycalc
    fit_manifolds = ["1-torus", "1-cube"]
    base = "fly_cv/fly1_trial1_light"
    Yind = 2
    comb = false
elseif T2
    fit_manifolds = ["2-torus", "2-cube"] #first term is data identity, second is model to compare
    base = "cv/"*fit_manifolds[1]
    Yind = 3
elseif SO3
    fit_manifolds = ["SO3", "3-cube"] #first term is data identity, second is model to compare
    base = "cv/"*fit_manifolds[1]
    Yind = 3
elseif threefold
    fit_manifolds = ["2-torus", "SO3", "3-sphere"]
    fit_manifolds = ["SO3", "3-sphere", "2-torus"]
    fit_manifolds = ["3-sphere", "2-torus", "SO3"]
    base = "cv/"*fit_manifolds[1]
    addstr = "_3fold"
    Yind = 3
else
    println("No method selected")
end

nmin = 1
niters = 10 #10 iterations
ext = ".png"

#store results
allerrs = zeros(2*niters, length(fit_manifolds))
allstds = zeros(2*niters, length(fit_manifolds))
ELBOs = zeros(niters, length(fit_manifolds))
LLs = zeros(niters, length(fit_manifolds))
tLLs = zeros(niters, length(fit_manifolds))

for niter = nmin:niters
    #println(niter)
    global allerrs, allstds
    for (m, fit_manifold) = enumerate(fit_manifolds) #consider each fitted manifold

        n, metric, ntilde = get_params(fit_manifold)
        _, _, _, _, _, _, dfunc = initialize_parameters(fit_manifold, zeros(1,1), 3, 1, 1)
        #load data
        fname = base*"_"*fit_manifold
        d = load("results/"*fname*addstr*string(niter)*".jld") #load data
        θfit=d["fit"]; θcv1=d["cv1"]; θcv2=d["cv2"]; true_params=d["true"]
        Xs1 = d["Xs1"]; Xs2 = d["Xs2"]; trainTs = d["fitTs"]; testTs = d["testTs"]
        ELBOs[niter, m] = d["Lfit"]
        Y = true_params[Yind]
        N, T = size(Y)

        #add diagonal component - this is done automatically in fit_mgplvm
        θfit[2] = [Σ + 1e-4*I for Σ = θfit[2]]
        θcv1[2] = [Σ + 1e-4*I for Σ = θcv1[2]]
        θcv2[2] = [Σ + 1e-4*I for Σ = θcv2[2]]

        #importance weighted LL for the training data
        LL = calc_LL(θfit, Y[:, trainTs], n, dfunc, m = 1000, comb = comb)
        LLs[niter, m] = LL

        newtLLs = [] #test LL
        for group = ["1", "2"]
            if group == "1" #specify what's train/test
                θtest = θcv1
                Xsnew = Xs2
            elseif group == "2"
                θtest = θcv2
                Xsnew = Xs1
            end

            errs = []
            nsamps = 50 #number of samples for computing MSE
            posterior = zeros(length(Xsnew), nsamps, length(testTs)) #number of neurons by number of samples by number of testpoints
            for j = 1:nsamps
                Xtildes = [randn(1, n) for i = 1:T]

                #this is where I think I was for training
                ChStrain = [cholesky(θfit[2][i]+1e-8*I) for i = 1:length(trainTs)]
                Xstrain = [Xtildes[t]*ChStrain[i].U for (i, t) = enumerate(trainTs)]
                gstrain = [expmap(X, dfunc) for X = Xstrain]
                gstrain = reduce(vcat, [translate_group(gstrain[i], θfit[1][i], dfunc) for i = 1:length(trainTs)])'

                #this is where I think I was for testing
                ChStest = [cholesky(θtest[2][t]+1e-8*I) for t = testTs]
                Xstest = [Xtildes[t]*ChStest[i].U for (i, t) = enumerate(testTs)]
                gstest = [expmap(X, dfunc) for X = Xstest]
                gstest = reduce(vcat, [translate_group(gstest[i], θtest[1][t], dfunc) for (i, t) = enumerate(testTs)])'

                for (ind, X) = enumerate(Xsnew) #look at the data we didn't fit
                    if flycalc
                        alphast, lst, sigst, us = θfit[3][X], θfit[4][X], θfit[5][X], θfit[6][X] #neuron
                    else
                        alphast, lst, sigst, us = θfit[3][1], θfit[4][1], θfit[5][1], θfit[6][1] #neuron
                    end
                    mus, covs = calc_sparse_posterior(gstest, gstrain, Y[X, trainTs], us, sigst, alphast, lst, dfunc) #prediction
                    samps = randn(1, length(mus))
                    samps = samps * cholesky(Symmetric(covs)+1e-6*I).U .+ mus' #sample from posterior
                    posterior[ind, j, :] = samps[:]
                end
            end

            if flycalc
                θsLL = [θtest[1][testTs], θtest[2][testTs], θfit[3][Xsnew], θfit[4][Xsnew], θfit[5][Xsnew], θfit[6][Xsnew]]
            else
                θsLL = [θtest[1][testTs], θtest[2][testTs], θfit[3], θfit[4], θfit[5], θfit[6]]
            end
            #compute test LL
            tLL = calc_LL(θsLL, Y[Xsnew, testTs], n, dfunc, m = 1000, comb = comb)
            newtLLs = [newtLLs; tLL]
            Ypred = mean(posterior, dims = 2)[:, 1, :]
            index = Int(niter + (parse(Int, group)-1)*niters)
            err = mean((Ypred .- Y[Xsnew, testTs]).^2)
            allerrs[index, m] = err
            println("new group ", fit_manifold, " ", niter, " ", group, ": ", err, " ", LL, " ", newtLLs[end])
        end
        tLLs[niter, m] = sum(newtLLs)
    end
end

#store results for further analysis
save("../results/"*base*"_cv_comparison"*addstr*".jld", "allerrs", allerrs, "ELBOs", ELBOs, "LLs", LLs, "tLLs", tLLs)
