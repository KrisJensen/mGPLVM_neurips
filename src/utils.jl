"""
assorted functions for plotting and computing some geometrical quantitites
"""

using PyPlot, SpecialFunctions, DistributionsAD
rc("font", family="sans-serif", size = 14)
rc("pdf", fonttype = 42)

function Ssphere(n; r=1) #surface area of n-sphere
    2*pi^((n+1)/2)/gamma((n+1)/2)*r^n
end

function Vball(n; r=1) #volume of n-ball
    pi^(n/2)/gamma(n/2+1)*r^n
end

function plot_activities(ys, fname; fsize = (10, 5), line = 0)
    #plot an activity matrix Y
    figure(figsize = fsize)
    imshow(ys, cmap = "Greys", aspect = "auto")
    if line > 0 axvline(line, ls = ":", color = "b") end
    savefig("figures/"*fname*".png", bbox_inches = "tight")
    close()
end

function get_params(manifold)
    #n, type, ntilde
    params = Dict([("1-sphere",(1, "theta", 1)), #circle
                    ("3-sphere",(3, "theta", 3)), #hypersphere
                    ("1-cube",(1, "euclid", 1)), #line
                    ("2-cube",(2, "euclid", 2)), #plane
                    ("3-cube",(3, "euclid", 3)), #cube
                    ("1-torus", (1, "torus", 1)), #ring
                    ("2-torus", (2, "torus", 2)), #torus
                    ("3-torus", (3, "torus", 3)), #hypertorus
                    ("SO3", (3, "SO3", 3)) #SO(3)
                    ])
    return params[manifold]
end

calc_X(Xtildes, μ, Σ) = Xtildes * cholesky(Σ+1e-6*I).U .+ μ'

function plot_torus(xs, cols; fname = "", r = 1, R = 1.6, alpha = 1, s = 10, x2s = [],
            ext = ".png", x2cols = "default", x2marks = "x", azim = 60, off2 = zeros(3))
    #makes a scatterplot of a torus with colors given by cols
    #xs is an Nx2 matrix of theta1, theta2
    x = (R .+ r*cos.(xs[:, 1])) .* cos.(xs[:, 2])
    y = (R .+ r*cos.(xs[:, 1])) .* sin.(xs[:, 2])
    z = r*sin.(xs[:, 1])
    if length(x2s) > 0
        x2 = (R .+ r*cos.(x2s[:, 1])) .* cos.(x2s[:, 2]) .+ off2[1]
        y2 = (R .+ r*cos.(x2s[:, 1])) .* sin.(x2s[:, 2]) .+ off2[2]
        z2 = r*sin.(x2s[:, 1]) .+ off2[3]
        if x2cols == "default" x2cols = cols end
    end
    figure(figsize = (6,6))
    scatter3D(x, y, z, marker = "o", c = cols, alpha = alpha, s= s, depthshade = true, edgecolors = "face", cmap = "Greys")
    if length(x2s) > 0
        scatter3D(x2, y2, z2, marker = x2marks, c = x2cols, alpha = 1, s= s, depthshade = true, edgecolors = "face")
    end
    plot3D(R*cos.(0:0.1:2*pi), R*sin.(0:0.1:2*pi), 0*(0:0.1:2*pi), "k-", alpha = 0.5)
    gca().view_init(elev=50, azim=azim)
    ax=gca(); ax.set_axis_off()
    zlim(-1.01*R, 1.01*R)
    xlim(-1.01*R, 1.01*R)
    ylim(-1.01*R, 1.01*R)
    savefig(fname*ext, bbox_inches = "tight")
    close()
end


function plot_solid_torus(θ, ϕ, cols; fname = "", r = 1, R = 1.6, alpha = 1, s = 10, x2s = [],
            ext = ".png", x2cols = "default", x2marks = "x", azim = 60, off2 = zeros(3),
            figsize = (6,6))
    #plots a torus surface plot with colors given by cols
    X = (R .+ r * cos.(ϕ)) .* cos.(θ)
    Y = (R .+ r * cos.(ϕ)) .* sin.(θ)
    Z = r * sin.(ϕ)
    figure(figsize = figsize)
    plot_surface(X, Y, Z, facecolors = cols, rstride = 1, cstride = 1)
    gca().view_init(elev=50, azim=azim)
    ax=gca(); ax.set_axis_off()
    zlim(-1.01*R, 1.01*R)
    xlim(-1.01*R, 1.01*R)
    ylim(-1.01*R, 1.01*R)
    savefig(fname*ext, bbox_inches = "tight", transparent = true)
    close()
end

function logmap_so3(qs)
    #logarithmic map for SO(3)
    qs = qs ./ sqrt.(sum(qs.^2, dims = 2)) #normalize to S3
    qs = sign.(qs[:, 1]) .* qs #double coverage
    θs = acos.(qs[:, 1]) #algebra norms (ϕ/2)
    us = qs[:, 2:4] #axes of rotation
    us = us ./ sin.(θs) #unit vectors
    vs = us .* θs
end
