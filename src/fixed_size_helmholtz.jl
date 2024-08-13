using CSV
using DataFrames
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
using SpecialFunctions

BLAS.set_num_threads(1)

#using HMatrices: laplace_matrix

include(joinpath(HMatrices.PROJECT_ROOT, "test", "testutils.jl"))


## Physical parameters
λ = 0.005 #
k = 2π / λ # wavenumber
θ = π / 4 # angle of incident wave

## Mesh parameters
meshsize = λ / 10 # mesh size
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
range_values = Vector{Float64}()
push!(range_values,Inf)

for i=2:Int(-log10(ε))
    push!(range_values,10.0^(-i))
end

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
    

    results_approx[k] = minimum(benchr_approx).time
    results_exact[k] = minimum(benchr_exact).time

    H_iprod.rtol = range_values[k]
    mul!(y_exact,L,g,1,0)
    mul!(y_approx,H_iprod,g,1,0)

    push!(rel_error_sol, norm(y_exact-y_approx)/norm(y_exact))
end



speed_up = (results_exact ) ./ results_approx

df = DataFrame("Rel. error" => rel_error_sol, "Speed up" => speed_up)
CSV.write("Helmholtz_product_results.csv", df)


