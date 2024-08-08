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
λ = 0.01 #
k = 2π / λ # wavenumber
θ = π / 4 # angle of incident wave

## Mesh parameters
meshsize = λ / 10 # mesh size, chose to have ~85% of compression
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
results_itn = zeros(length(range_values))

rel_error_sol = Vector{Float64}()


y_exact = similar(g)
y_approx = similar(g)

for i=1:length(range_values)
    benchr_approx = @benchmark igmres($L,$g,tol=range_values[$i]) 
    benchr_exact = @benchmark InexactGMRES.test_gmres($L,$g,tol=range_values[$i])
    #benchr_exact = @benchmark gmres($L,$g;reltol=range_values[$i])
    

    results_approx[i] = minimum(benchr_approx).time
    results_exact[i] = minimum(benchr_exact).time

  
    
    y_exact = gmres(L,g;reltol=range_values[i])
    
    y_approx,res_aprox,it = igmres(L,g,tol=range_values[i])

    results_itn[i] = it
    push!(rel_error_sol, norm(L*y_approx - g)/norm(g))
end



speed_up = (results_exact ) ./ results_approx

df = DataFrame("Rel. error" => rel_error_sol, "Speed up" => speed_up)
CSV.write("Helmholtz_gmres_results.csv", df)


# p1 = Plots.scatter(range_values, rel_error_sol, title="Relative error between solutions",legend=false,yaxis=:log,xaxis=:log)
# Plots.xlabel!(p1,"Overall Tolerance σ")
# Plots.ylabel!(p1,"Relative Error")
# Plots.ylims!(p1,10.0^(-10),10.0^(-1))
# Plots.yticks!(p1,range_values)
# Plots.xticks!(p1,range_values)


# p2 = Plots.plot(range_values,speed_up, title="Speed up", legend=false,xaxis=:log)
# Plots.xlabel!(p2,"Overall Tolerance σ")
# Plots.ylabel!(p2,"Speed up")
# Plots.xticks!(p2,range_values)


# p3 = Plots.plot(range_values,results_itn, title="Number of Iterations", legend=false,xaxis=:log)
# Plots.xlabel!(p3,"Overall Tolerance σ")
# Plots.ylabel!(p3,"Iterations")
# Plots.xticks!(p3,range_values)

######
#Rel error between solutions x Matrix size and residual x iteration number

# bigger_size = length(res_exact) > length(res_approx) ? length(res_approx) : length(res_exact)

# Plots.plot(p1,p2,layout = (2,1))