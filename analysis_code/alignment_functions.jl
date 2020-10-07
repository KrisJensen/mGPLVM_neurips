"""
code for aligning latent means while preserving pairwise distances.
used for visualizations.
"""

using Zygote, Statistics
include(joinpath(@__DIR__, "../src/", "translation_functions.jl"))

function align_torus(thetas, target)
    #code for aligning the result of a 2-torus calculation
    opt(offset1, offset2, alpha1, alpha2) = mean(1 .- cos.(([alpha1; alpha2] .* thetas .+ [offset1; offset2] .- target)))
    alpha1s = [-1.; -1.; 1.; 1.]
    alpha2s = [-1.; 1.; -1.; 1.]
    losses = [0.; 0.; 0.; 0.]
    offset1s = [0.; 0.; 0.; 0.]
    offset2s = [0.; 0.; 0.; 0.]
    for (i, alpha1) = enumerate(alpha1s)
        alpha2 = alpha2s[i]
        println("alpha1 ", alpha1, " alpha2 ", alpha2)
        offset1, offset2 = 0, 0
        for n = 1:300
            g = gradient(opt, offset1, offset2, alpha1, alpha2)
            offset1 -= 1e-1 * g[1]
            offset2 -= 1e-1 * g[2]
            if n % 50 == 0 println(offset1, " ", offset2, " ", opt(offset1, offset2, alpha1, alpha2)) end
        end
        if offset1 < 0 offset1 += 2*pi end
        if offset2 < 0 offset2 += 2*pi end
        offset1s[i] = offset1
        offset2s[i] = offset2
        losses[i] = opt(offset1, offset2, alpha1, alpha2)
    end
    return alpha1s[argmin(losses)], alpha2s[argmin(losses)], offset1s[argmin(losses)], offset2s[argmin(losses)]
end

function align_theta(thetas, target; Print = true)
    #code for aligning the result of a 1-torus calculation
    opt(offset, alpha) = mean(1 .- cos.((alpha*thetas .+ offset - target)))
    alphas = [-1.; 1.]
    losses = [0.; 0.]
    offsets = [0.; 0.]
    for (i, alpha) = enumerate(alphas)
        Print && println("alpha ", alpha)
        offset = 0
        for n = 1:300
            g = gradient(opt, offset, alpha)
            offset -= 1e-1 * g[1]
            Print && if n % 50 == 0 println(offset, " ", opt(offset, alpha)) end
        end
        if offset < 0 offset += 2*pi end
        offsets[i] = offset
        losses[i] = opt(offset, alpha)
    end
    return alphas[argmin(losses)], offsets[argmin(losses)]
end
