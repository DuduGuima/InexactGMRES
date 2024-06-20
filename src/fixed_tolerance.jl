using Test
using HMatrices
using LinearAlgebra
using Random
using StaticArrays
using BenchmarkTools
using HMatrices: RkMatrix
using LoopVectorization
using IterativeSolvers
using InexactGMRES
using Plots

#using HMatrices: laplace_matrix




include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))

n_points=7

range_values= [2^i for i in 10:(10 + n_points)]

results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))

rel_error_sol = Vector{Float64}()
n_iterations_exact = Vector{Float64}()
n_iterations_approx = Vector{Float64}()

const EPS = 1e-8
const Point2D = SVector{2,Float64}

function laplace_matrix(X, Y)
    f = (x, y) -> begin
        d = norm(x - y) + EPS
        inv(2π)*log(d)
    end
    return KernelMatrix(f, X, Y)
end


function helmholtz_matrix(X, Y, k=0.5)
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
    X = Y = [Point2D(sin(i*2π/n),cos(i*2π/n)) for i in 0:n-1]
    nothing

   
    splitter = CardinalitySplitter(; nmax = 50)
    Xclt = ClusterTree(X, splitter)
    Yclt = ClusterTree(Y, splitter)
    adm = StrongAdmissibilityStd(; eta = 3)
    atol = 1e-6
    comp = PartialACA(; atol)

    #K = laplace_matrix(X, Y)
    K = helmholtz_matrix(X,Y)

    H = assemble_hmatrix(K,Xclt,Yclt;adm,comp,threads=false,distributed=false)
    T = eltype(H)   
   
    b=rand(T,n) 
    # y = similar(b)

    # benchr_exact = @benchmark gmres($H,b) setup=(b=rand($T,$n));
    # benchr_exact = @benchmark InexactGMRES.test_gmres($H,$b,tol=1e-5) 
    # benchr_approx = @benchmark igmres($H,$b,tol=1e-5) 

    # results_approx[k] = mean(benchr_approx).time
    # results_exact[k] = mean(benchr_exact).time

    #y_exact= gmres(H,b;reltol=1e-5)
    y_exact,res_exact,n_it_exact = InexactGMRES.test_gmres(H,b,tol=atol)
    y_approx,res_approx,n_it_approx = igmres(H,b,tol=atol)

    push!(n_iterations_exact,n_it_exact)
    push!(n_iterations_approx,n_it_approx)
    push!(rel_error_sol, norm(y_exact-y_approx)/norm(y_exact))
end



# speed_up = (results_exact ) ./ results_approx


# p1 = plot(range_values,[results_approx results_exact].*1e-9,title="Total execution time",label=["Inexact GMRES" "Exact GMRES"],legend=true)
# xlabel!(p1,"Matrix size")
# ylabel!(p1,"Ex. time[ns]")


# p2 = plot(range_values,speed_up, title="Speed up", legend=false)
# xlabel!(p2,"Matrix size")
# ylabel!(p2,"Speed up")


######
#Rel error between solutions x Matrix size and residual x iteration number

# bigger_size = length(res_exact) > length(res_approx) ? length(res_approx) : length(res_exact)

p1 = scatter(range_values, rel_error_sol, title="Relative error between exact and inexact solution",legend=false,yaxis=:log)
xlabel!(p1,"Matrix Size")
ylabel!(p1,"Relative Error")


p2 = plot(range_values,[n_iterations_approx n_iterations_exact],label= ["Inexact GMRES" "Exact GMRES"], title="Number of Iterations",legend=true)
xlabel!(p2,"Matrix Size")
ylabel!(p2,"Iterations")


plot(p1,p2, layout=(2,1))
