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
using SpecialFunctions

#using HMatrices: laplace_matrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))


## Physical parameters
λ = 0.25 #
k = 2π / λ # wavenumber
θ = π / 4 # angle of incident wave

## Mesh parameters
meshsize = λ / 600 # mesh size
gorder = 2 # polynomial order for geometry
qorder = 4 # quadrature order

#creating circle

circle = Inti.parametric_curve(0.0,1.0;labels=["circle"]) do s
    return SVector(cos(2π*s[1]),sin(2π*s[1]))
end
Γ = circle

msh=Inti.meshgen(Γ; meshsize)

Q = Inti.Quadrature(msh; qorder)

## The incident field
d = SVector(cos(θ), sin(θ)) # incident direction
uᵢ = (x) -> exp(im * k * dot(x, d)) # incident plane wave


#making the kernel
function helmholtz_custom(target, source, k)
    x,y,ny = Inti.coords(target),Inti.coords(source),Inti.normal(source)
    r = x-y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = im / 4 * hankelh1(0, k * d)
    dod_term = (im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny))
    return filter*(dod_term - im*k*sod_term)
end

K = let k = k
    (t,q) -> helmholtz_custom(t,q,k)
end

## Right-hand side given by Dirichlet trace of plane wave
g = map(Q) do q
    # normal derivative of e^{ik*d⃗⋅x}
    x, ν = q.coords, q.normal
    return -uᵢ(x)
end ## Neumann trace on boundary


ε=1e-8#initial aproximation to hmatrix

Lop = Inti.IntegralOperator(K,Q,Q)
L = Inti.assemble_hmatrix(Lop; rtol = ε)
Id = sparse((0.5 + 0*im)I,size(L))
axpy!(1.0,Id,L) # in place sum of L

#arrays to store results
range_values= [10.0^(-i) for i in 2:Int(-log10(ε))]


results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))

rel_error_sol = Vector{Float64}()


y_exact = similar(g)
y_approx = similar(g)

H_iprod = HMatrices.ITerm(L,0.0)
for k=1:length(range_values)
    H_iprod.rtol = range_values[k]
    benchr_approx = @benchmark mul!($y_approx,$H_iprod,$g,1,0) 

    H_iprod.rtol = 0
    benchr_exact = @benchmark mul!($y_exact,$H_iprod,$g,1,0) 
    

    results_approx[k] = mean(benchr_approx).time
    results_exact[k] = mean(benchr_exact).time

    H_iprod.rtol = range_values[k]
    mul!(y_exact,L,g,1,0)
    mul!(y_approx,H_iprod,g,1,0)

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
