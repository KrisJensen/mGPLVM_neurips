"""
assorted functions for generating parametric tuning curves of the form
y = exp( -d_geo(g, g')^2 / 2*l^2 )
"""

function dtheta(xs, ps)
    #S^n: theta = cos^-1(dot(g, g'))
    ds = (xs./sqrt.(sum(xs.^2, dims = 2))) * (ps./sqrt.(sum(ps.^2, dims = 1)))
    ds = acos.( ds*0.99999999 )
end

function deuclid(xs, ps)
    #R^n: ||x-y||
    xsr = reshape(xs, size(xs, 1), size(xs, 2), 1)
    psr = reshape(ps, 1, size(ps, 1), size(ps, 2))
    ds = sqrt.(sum((xsr .-psr).^2, dims = 2)[:, 1, :])
end

function dtorus(xs, ps)
    #T^n: || theta_1 - theta_2 || along shortest path
    xsr = reshape(xs, size(xs, 1), size(xs, 2), 1)
    psr = reshape(ps, 1, size(ps, 1), size(ps, 2))
    ds = acos.(cos.(xsr .- psr)*0.999999999)
    ds = sqrt.(sum(ds.^2, dims = 2)[:, 1, :])
end

function dso3(xs, ps)
    #SO(3): theta_rot
    ds = (xs./sqrt.(sum(xs.^2, dims = 2))) * (ps./sqrt.(sum(ps.^2, dims = 1))) #dot product of unit quarternions
    ds = 2*acos.( abs.(ds)*0.99999999 ) #q*qj = cos(theta/2) defined up to a sign
end

function calcg(xs, ps, ls, alphas, betas, gammas; dfunc = dtheta, noised = false)
    #'Gaussian bump'-tuning curves in the geodesic distance
    ds = dfunc(xs, ps)
    if noised println(mean(abs.(ds))/10); ds = ds + randn(size(ds))*mean(abs.(ds))/8 end
    ys = alphas .* exp.( -ds.^2 ./ (2*ls.^2) ) .+ betas #NxT expected activities
    return gammas' .* ys #scale by global activity
end
