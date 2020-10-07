"""
functions for shifting the mean of a distribution on a manifold
for mean g^mu,
g = g^mu g_tilde
"""

function translate_euclid(Xs, μ)
    #add mean
    #Xs is mxn, mu is n
    return Xs .+ μ'
end

function translate_torus(Xs, μ)
    #add mean and let covariance function take care of the mod 2pi
    #Xs is mxn, mu is n
    return Xs .+ μ'
end

function qproduct(p, q)
    return [
    p[1]*q[1] - p[2]*q[2] - p[3]*q[3] - p[4]*q[4];
    p[1]*q[2] + p[2]*q[1] + p[3]*q[4] - p[4]*q[3];
    p[1]*q[3] - p[2]*q[4] + p[3]*q[1] + p[4]*q[2];
    p[1]*q[4] + p[2]*q[3] - p[3]*q[2] + p[4]*q[1]
    ]
end

function qproducts(p, q)
    #p is length 4
    #q is mx4
    #p = p/norm(p)
    r1 = p[1]*q[:,1] - p[2]*q[:,2] - p[3]*q[:,3] - p[4]*q[:,4]
    r2 = p[1]*q[:,2] + p[2]*q[:,1] + p[3]*q[:,4] - p[4]*q[:,3]
    r3 = p[1]*q[:,3] - p[2]*q[:,4] + p[3]*q[:,1] + p[4]*q[:,2]
    r4 = p[1]*q[:,4] + p[2]*q[:,3] - p[3]*q[:,2] + p[4]*q[:,1]
    return [r1 r2 r3 r4]
end

function translate_so3(qs, μ)
    #left-multiply each X by a quaternion μ
    #Xs is mxn, mu is n
    return qproducts(μ, qs)
end

function translate_s3(qs, μ)
    #left-multiply each X by a quaternion μ
    #Xs is mxn, mu is n
    return qproducts(μ, qs)
end
