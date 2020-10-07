"""
functions for computing distances, kernels, and p(Y|X) for different manifolds
"""

using LinearAlgebra

function deuclid_gp(xs, ps)
    #Euclidean distance function ||x-y||
    xsr = reshape(xs, size(xs, 1), size(xs, 2), 1)
    psr = reshape(ps, 1, size(ps, 1), size(ps, 2))
    ds = sum((xsr .-psr).^2, dims = 2)[:, 1, :] #||x-x'||_2^2
end

function dtorus_gp(xs, ps)
    #toroidal distance 2*sum(1 - cos(g-g'))
    xsr = reshape(xs, size(xs, 1), size(xs, 2), 1)
    psr = reshape(ps, 1, size(ps, 1), size(ps, 2))
    ds = sum(2 .- 2*cos.(xsr .- psr), dims = 2)[:, 1, :] #sum_i{ 2 - 2*cos(x_i - x_i') } \approx sum_i{(x_i-x_i')^2}
end

function dso3_gp(xs, ps)
    #SO(3) distance function 4*(1 - dot(g, g')^2)
    ds = (xs./sqrt.(sum(xs.^2, dims = 2))) * (ps./sqrt.(sum(ps.^2, dims = 1))) #dot product of unit quarternions
    ds = 2*ds.^2 .- 1 #cos(theta_rot)
    ds = 2 .- 2*ds #2 - 2 cos(theta_rot)
end

function dtheta_gp(xs, ps)
    #S^n distance function 2*(1 - dot(g, g'))
    ds = (xs./sqrt.(sum(xs.^2, dims = 2))) * (ps./sqrt.(sum(ps.^2, dims = 1))) #cos theta
    ds = 2 .- 2*ds #2 - 2*cos(theta) \approx theta^2 \approx sum_i{(x_i-x_i')^2}
end

function calc_K(xs, ys, alpha, l, dfunc)
    #compute kernel
    d = dfunc(xs', ys) #O(m^2)
    K =  alpha * exp.( -d / (2*l^2) ) #O(N m^2)
    return K
end

function calc_GP_L(ys, ps, alphas, ls, sigs, dfunc)
    #compute logP(Y|X) without sparse approximation
    N, T = size(ys)
    Ks = [calc_K(ps, ps, alphas[i], ls[i], dfunc) for i = 1:N]
    ChKs = [cholesky(K+ sig^2*I + 1e-8*I) for K = Ks]

    alphas = [ChKs[i].U \ (ChKs[i].L \ ys[i, :]) for i = 1:N] #rasmussen and williams p19
    Ls = [-0.5*ys[i, :]'*alphas[i] - sum(log.(diag(ChKs[i].L))) - T/2*log(2*pi) for i = 1:N]#rasmussen and williams p19
    Ls = sum(Ls)

    return Ls
end

function calc_sparse_GP_L(Y, X, alphas, ls, sigs, us, dfunc)
    #compute logP(Y|X) with sparse approximation
    #us are the inducing points
    #Y are observations (NxT)
    #X are latents (nxT)

    N, T = size(Y)
    Kfu =  [calc_K(X, us[i], alphas[i], ls[i], dfunc) for i = 1:N] #O(N T m) #private inducing points per neuon us[i]
    Kuu = [calc_K(us[i], us[i], alphas[i], ls[i], dfunc) for i = 1:N]  #O(N m^2)

    #woodbury rules from Rasmussen 201
    Z = sigs.^2 #make this more exciting for other approximations (e.g. sig^2 I + diag(Kff - Qff) for FITC)
    Zinv = sigs.^(-2) #O(N)

    F = [Symmetric(Kuu[i] + Kfu[i]' * Zinv[i] * Kfu[i]) for i = 1:N] #O(N m^2 T ) (Z is diagonal)
    ChF = [cholesky(F[i] + 1e-6*I) for i = 1:N] #O(N m^3)

    alpha = [ChF[i].U \ (ChF[i].L \ (Kfu[i]' * Zinv[i]) ) for i = 1:N] #O(m^2 N)
    K_inv =  [Zinv[i]*I - Zinv[i]*Kfu[i]*alpha[i] for i = 1:N] #O(N T^2 m ) (???) #this is the inverse of Q_ff + G = Z + U*W*Vt

    ChK = [cholesky(Kuu[i] + 1e-6*I) for i = 1:N] #O(m^3)
    logZ = T*log.(sigs.^2) #need this if we don't do something more fancy
    logK = [logZ[i] - 2*sum(log.(diag(ChK[i].L))) + 2*sum(log.(diag(ChF[i].L))) for i = 1:N] #log|Kuu^-1| = log|Kuu|^-1 = -log|Kuu|
    L = [ (-0.5*(Y[i, :]'*K_inv[i]*Y[i, :])[1] -0.5*logK[i] -T/2*log(2*pi)) for i = 1:N] #O(N T^2)

    alpha = [ChK[i].U \ (ChK[i].L \ Kfu[i]') for i = 1:N]
    Tr_Qff = [tr(Kfu[i] * alpha[i]) for i = 1:N]
    Tr_Kff = [T*alphas[i] for i = 1:N] #only need the trace; the exp term is 1 so k(x,x) = alpha
    L = [L[i] - 1/(2*sigs[i]^2) * ( Tr_Kff[i] - Tr_Qff[i] ) for i = 1:N] #add this for titsias 2009 approximation

    return sum(L)
end


function calc_sparse_GP_L_comb(Y, X, alphas, ls, sigs, us, dfunc)
    #compute logP(Y|X) with sparse approximation and shared hyperparameters
    #us are the inducing points
    #Y are observations (NxT)
    #X are latents (nxT)
    #combined tuning curves

    N, T = size(Y)
    Kfu =  calc_K(X, us[1], alphas[1], ls[1], dfunc) #O(N T m)
    Kuu = calc_K(us[1], us[1], alphas[1], ls[1], dfunc)  #O(N m^2)

    #woodbury rules from Rasmussen 201
    Z = sigs[1]^2 #make this more exciting for other approximations (e.g. sig^2 I + diag(Kff - Qff) for FITC)
    Zinv = sigs[1]^(-2) #O(N)

    F = Symmetric(Kuu + Kfu' * Zinv * Kfu) #O(N m^2 T ) (Z is diagonal)
    ChF = cholesky(F + 1e-6*I) #O(N m^3)

    alpha = ChF.U \ (ChF.L \ (Kfu' * Zinv) ) #O(m^2 N)
    K_inv =  Zinv*I - Zinv*Kfu*alpha #O(N T^2 m ) (???) #this is the inverse of Q_ff + G = Z + U*W*Vt

    ChK = cholesky(Kuu + 1e-6*I) #O(m^3)
    logZ = T*log(sigs[1]^2) #need this if we don't do something more fancy
    logK = logZ - 2*sum(log.(diag(ChK.L))) + 2*sum(log.(diag(ChF.L))) #log|Kuu^-1| = log|Kuu|^-1 = -log|Kuu|
    L = (-0.5*sum(Y' .* (K_inv*Y'), dims = 1)[:] .-0.5*logK .-T/2*log(2*pi))

    alpha = ChK.U \ (ChK.L \ Kfu')
    Tr_Qff = tr(Kfu * alpha)
    Tr_Kff = T*alphas[1] #only need the trace
    L = L .- 1/(2*sigs[1]^2) * ( Tr_Kff - Tr_Qff ) #add this for titsias 2009 approximation

    return sum(L)
end
