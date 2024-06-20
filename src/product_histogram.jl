using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using BenchmarkTools
using HMatrices: RkMatrix
using LoopVectorization
using InexactGMRES
using Plots

#using HMatrices: laplace_matrix




include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

n_points=5

range_values= [2^i for i in 10:(10 + n_points)]

results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))

const EPS = 1e-8
const Point2D = SVector{2,Float64}

function laplace_matrix(X, Y)
    f = (x, y) -> begin
        d = norm(x - y) + EPS
        inv(4π * d)
    end
    return KernelMatrix(f, X, Y)
end


function helmholtz_matrix(X, Y, k)
    f = (x, y) -> begin
        EPS = 1e-8 # fudge factor to avoid division by zero
        d = norm(x - y) + EPS
        exp(im * k * d) * inv(4π * d)
    end
    return KernelMatrix(f, X, Y)
end

function elastostatic_matrix(X, Y, μ, λ)
    f = (x, y) -> begin
        ν = λ / (2 * (μ + λ))
        r = x - y
        d = norm(r) + EPS
        RRT = r * transpose(r) # r ⊗ rᵗ
        return 1 / (16π * μ * (1 - ν) * d) * ((3 - 4 * ν) * I + RRT / d^2)
    end
    return KernelMatrix(f, X, Y)
end


for k=1:length(range_values)
    # points on a circle
    m = n = range_values[k]
    X = rand(SVector{3,Float64}, m)
    Y = [rand(SVector{3,Float64}) for _ in 1:n]
    splitter = CardinalitySplitter(; nmax = 50)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    atol = 1e-5
    comp = PartialACA(; atol)

    
    
    # create the abstract matrix
    K = laplace_matrix(X, Y)
   

    H = assemble_hmatrix(K,Xclt,Yclt;adm,comp,threads=false,distributed=false)
    T = eltype(H)
    b=rand(T,n) 
    y = similar(b)

    benchr_exact = @benchmark mul!($y,$H,b,1,0;threads=false) setup=(b=rand($T,$n); y = similar(b));
    benchr_approx = @benchmark mul!($y,$H,b,1,0,1e-5;threads=false) setup=(b=rand($T,$n); y = similar(b));

    results_approx[k] = mean(benchr_approx).time
    results_exact[k] = mean(benchr_exact).time

end

speed_up = (results_exact) ./ results_approx


p1 = plot(range_values,[results_approx results_exact].*1e-9,title="Total execution time",label=["Approx. product" "Exact product"])

ylabel!(p1,"Ex. time[ns]")


p2 = plot(range_values,speed_up, title="Speed up", legend=false)
xlabel!(p2,"Matrix size")
ylabel!(p2,"Speed up")



plot(p1,p2, layout=(2,1))
