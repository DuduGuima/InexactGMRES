using Test
using Inti
using HMatrices
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using BenchmarkTools
using HMatrices: RkMatrix
using LoopVectorization
using IterativeSolvers
using InexactGMRES
using Plots

#using HMatrices: laplace_matrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))


#creating circle

circle = Inti.parametric_curve(0.0,1.0;labels=["circle"]) do s
    return SVector(cos(2π*s[1]),sin(2π*s[1]))
end
Γ = circle
size_mesh = 2^13
msh=Inti.meshgen(Γ,size_mesh)

Q = Inti.Quadrature(msh; qorder=4)


#making the kernel
function laplace(target, source)
    x,y,ny = Inti.coords(target),Inti.coords(source),Inti.normal(source)
    r = x-y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = -1 / (2π) *log(d)
    #dod_term = 1 / (2π) / (d^2) * dot(r, ny)
    return filter*(sod_term)
end

K = let 
    (t,q) -> laplace(t,q)
end

ε=1e-8#initial aproximation to hmatrix

Lop = Inti.IntegralOperator(K,Q,Q)
L = Inti.assemble_hmatrix(Lop; rtol = ε)
Id = sparse((-0.5 + 0*im)I,size(L))
axpy!(-1.0,Id,L) # in place sum of L

#arrays to store results
range_values= [10.0^(-i) for i in 2:Int(-log10(ε))]


results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))

rel_error_sol = Vector{Float64}()

T = eltype(L)
b=rand(T,length(Q))
y_exact = similar(b)
y_approx = similar(b)

H_iprod = HMatrices.ITerm(L,0.0)
for k=1:length(range_values)
    H_iprod.rtol = range_values[k]
    benchr_approx = @benchmark mul!($y_approx,$H_iprod,$b,1,0) 

    H_iprod.rtol = 0
    benchr_exact = @benchmark mul!($y_exact,$H_iprod,$b,1,0) 
    

    results_approx[k] = mean(benchr_approx).time
    results_exact[k] = mean(benchr_exact).time

    H_iprod.rtol = range_values[k]
    mul!(y_exact,L,b,1,0)
    mul!(y_approx,H_iprod,b,1,0)

    push!(rel_error_sol, norm(y_exact-y_approx)/norm(y_exact))
end



speed_up = (results_exact ) ./ results_approx


p1 = Plots.scatter(range_values, rel_error_sol, title="Relative error between products",legend=false,yaxis=:log,xaxis=:log)
Plots.xlabel!(p1,"Product Tolerance σ")
Plots.ylabel!(p1,"Relative Error")
Plots.ylims!(p1,10.0^(-10),10.0^(-1))
Plots.yticks!(p1,range_values)
Plots.xticks!(p1,range_values)


p2 = Plots.plot(range_values,speed_up, title="Speed up", legend=false,xaxis=:log)
Plots.xlabel!(p2,"Product Tolerance σ")
Plots.ylabel!(p2,"Speed up")
Plots.xticks!(p2,range_values)

######
#Rel error between solutions x Matrix size and residual x iteration number

# bigger_size = length(res_exact) > length(res_approx) ? length(res_approx) : length(res_exact)

Plots.plot(p1,p2,layout = (2,1))
