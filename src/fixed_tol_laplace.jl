using Inti
Inti.stack_weakdeps_env!(; update = false)

#-

## Load the necessary packages
using CSV
using DataFrames
using StaticArrays
using LinearAlgebra
using Meshes
using Gmsh
#using GLMakie
using HMatrices
using IterativeSolvers
using LinearMaps
using SpecialFunctions
using SparseArrays
using BenchmarkTools
#-

#Inexact
using InexactGMRES

BLAS.set_num_threads(1)

#range_values = LinRange(0.1,0.0025,7)
range_values = [0.25,0.1,0.05,0.025,0.01,0.005]
#arrays to store results


results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))
range_sizes = zeros(length(range_values))

rel_error_sol = Vector{Float64}()

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

circle = Inti.parametric_curve(0.0,1.0;labels=["circle"]) do s
    return SVector(cos(2π*s[1]),sin(2π*s[1]))
end
Γ = circle
println("Current progress: ",repeat(" ",32),"0%")
for i in eachindex(range_values)
    ## Physical parameters
    λ = range_values[i]
    k = 2π / λ # wavenumber
    θ = π / 4 # angle of incident wave


    ## Mesh parameters
    meshsize = λ / 10 # mesh size
    gorder = 2 # polynomial order for geometry
    qorder = 4 # quadrature order
    #creating circle

    

    msh=Inti.meshgen(Γ; meshsize)

    Q = Inti.Quadrature(msh; qorder)

    range_sizes[i] = length(Q)

    ## Use GMRES to solve the linear system
    ε = 1e-8
    σ = 1e-2
    Lop = Inti.IntegralOperator(K,Q,Q)
    L = Inti.assemble_hmatrix(Lop; rtol = ε)
    Id = sparse((0.5 + 0*im)I,size(L))
    axpy!(1.0,Id,L) # in place sum of L



    T = eltype(L)
    b = rand(T, length(Q))
    y_exact = similar(b)
    y_approx = similar(b)


    H_iprod = HMatrices.ITerm(L, 0.0)
    H_iprod.rtol=σ

    α=1
    β=0

    benchr_approx = @benchmark mul!($y_approx, $H_iprod, $b, $α, $β)
    benchr_exact = @benchmark mul!($y_exact, $H_iprod.hmatrix, $b, $α, $β)
    

    results_exact[i] = minimum(benchr_exact).time
    results_approx[i] = minimum(benchr_approx).time

    
    push!(rel_error_sol, norm(y_exact - y_approx) / norm(y_exact))
    marker = i/length(range_values)
    #println("Current % of measurement: ",100*(i/length(range_values)))
    println("Current progress: ",repeat("|",Int(round(marker*30))),repeat(" ",32-Int(round(marker*30))),round(100*marker,digits=1),"%")

end
# mul!(y_approx, H_iprod.hmatrix, g, 1, 0)
# mul!(y_exact, L, g, 1, 0)

# @assert y_exact == y_approx 
# ##
# b1 = @benchmark mul!($y_approx, $H_iprod, $g, 1, 0)
# b2 = @benchmark mul!($y_exact, $L, $g, 1, 0)
# ##

# H_iprod.rtol=ε

# b3 = @benchmark mul!($y_approx, $H_iprod, $g, 1, 0)

# H_iprod.rtol=Inf

# b4 = @benchmark mul!($y_approx, $H_iprod, $g, 1, 0)

# H_iprod.rtol=1e-6
# b5 = @benchmark mul!($y_approx, $H_iprod, $g, 1, 0)

##




speed_up = (results_exact ) ./ results_approx

df = DataFrame("Rel. error" => rel_error_sol, "Speed up" => speed_up, "Sizes"=>range_sizes)
CSV.write("Fixtol_laplace_results.csv", df)
