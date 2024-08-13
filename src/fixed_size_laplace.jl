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

circle = Inti.parametric_curve(0.0, 1.0; labels=["circle"]) do s
    return SVector(cos(2π * s[1]), sin(2π * s[1]))
end
Γ = circle
msh = Inti.meshgen(Γ; meshsize)

Q = Inti.Quadrature(msh; qorder=4)


#making the kernel
function laplace(target, source)
    x, y, ny = Inti.coords(target), Inti.coords(source), Inti.normal(source)
    r = x - y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = -1 / (2π) * log(d)
    #dod_term = 1 / (2π) / (d^2) * dot(r, ny)
    return filter * (sod_term)
end

K = let
    (t, q) -> laplace(t, q)
end

ε = 1e-8#initial aproximation to hmatrix

Lop = Inti.IntegralOperator(K, Q, Q)
L = Inti.assemble_hmatrix(Lop; rtol=ε)
Id = sparse((-0.5 + 0 * im)I, size(L))
axpy!(-1.0, Id, L) # in place sum of L

#arrays to store results
range_values = Vector{Float64}()
push!(range_values,Inf)

for i=2:Int(-log10(ε))
    push!(range_values,10.0^(-i))
end

results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))

rel_error_sol = Vector{Float64}()

T = eltype(L)
b = rand(T, length(Q))
y_exact = similar(b)
y_approx = similar(b)

H_iprod = HMatrices.ITerm(L, 0.0)

# mul!(y_approx, H_iprod.hmatrix, b, 1, 0)
# mul!(y_exact, L, b, 1, 0)

# @assert y_exact == y_approx 
# ##
# b1 = @benchmark mul!($y_approx, $H_iprod, $b, 1, 0)
# b2 = @benchmark mul!($y_exact, $L, $b, 1, 0)
# ##

# H_iprod.rtol=ε

# b3 = @benchmark mul!($y_approx, $H_iprod, $b, 1, 0)

# H_iprod.rtol=Inf

# b4 = @benchmark mul!($y_approx, $H_iprod, $b, 1, 0)

# H_iprod.rtol=1e-6
# b5 = @benchmark mul!($y_approx, $H_iprod, $b, 1, 0)

##
for k = 1:length(range_values)
    H_iprod.rtol = range_values[k]
    benchr_approx = @benchmark mul!($y_approx, $H_iprod, $b, 1, 0)

    H_iprod.rtol = 0
    benchr_exact = @benchmark mul!($y_exact, $H_iprod.hmatrix, $b, 1, 0)

    #utiliser le minium au lieu de la moyenne

    results_approx[k] = minimum(benchr_approx).time #utilizer min + variance
    results_exact[k] = minimum(benchr_exact).time
    #comparer les produits avec un produit avec une Hmatrice ou les blocs low rank = 0
    H_iprod.rtol = range_values[k]
    mul!(y_exact, L, b, 1, 0)
    mul!(y_approx, H_iprod, b, 1, 0)

    push!(rel_error_sol, norm(y_exact - y_approx) / norm(y_exact))
end



speed_up = (results_exact) ./ results_approx

df = DataFrame("Rel. error" => rel_error_sol, "Speed up" => speed_up)
CSV.write("Laplace_results.csv", df)

