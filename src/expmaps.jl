"""
exponential maps (Exp_G : R^n -> G)
g = Exp_G x
"""

function expmap_euclid(Xs)
    #exponential map is the identity
    return Xs
end

function expmap_torus(Xs)
    #expmap is mod 2pi but we can just let the covariance function take care of that
    return Xs
end


function expmap_so3(Xs)
    #exponential map for the quaternion
    θs = sqrt.(sum(Xs.^2, dims = 2)) #rotation angles are twice the norm (θ = ϕ/2)
    us = Xs ./ θs #axes of rotation are unit vectors
    qs = [cos.(θs) sin.(θs) .* us] #project onto group.
    #we could remove the degeneracy by making all qs have a positive first coordinate
    #but this is handled by the covariance function anyways (function of dot(g, g')^2)
    return qs
end

function expmap_s3(Xs)
    #exponential map for the quaternion
    θs = sqrt.(sum(Xs.^2, dims = 2)) #rotation angles are twice the norm (θ = ϕ/2)
    us = Xs ./ θs #axes of rotation are unit vectors
    qs = [cos.(θs) sin.(θs) .* us] #project onto group
    return qs
end
