
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

range_values = [0.25,0.1,0.05,0.025,0.01]

#arrays to store results

range_sizes = zeros(length(range_values))
results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))


results_itn = zeros(length(range_values))
rel_error_sol = Vector{Float64}()

##Definig custom kernel to make L in a single calculation(and leave it as a hmatrix)
function helmholtz_custom(target, source, k)
    x,y,ny = Inti.coords(target),Inti.coords(source),Inti.normal(source)
    r = x-y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = im / 4 * hankelh1(0, k * d)
    dod_term = (im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny))
    return filter*(dod_term - im*k*sod_term)
end


for i in eachindex(range_values)
    ## Physical parameters
    λ = range_values[i]
    k = 2π / λ # wavenumber
    θ = π / 4 # angle of incident wave

    K = let k = k
        (t,q) -> helmholtz_custom(t,q,k)
    end

    ## Mesh parameters
    meshsize = λ / 10 # mesh size
    gorder = 2 # polynomial order for geometry
    qorder = 4 # quadrature order

    ## Import the mesh
    filename = joinpath(Inti.PROJECT_ROOT, "docs", "assets", "elliptic_cavity_2D.geo")
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.open(filename)
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(gorder)
    msh = Inti.import_mesh(; dim = 2)
    gmsh.finalize()

    ## Extract the entities and elements of interest
    ents = Inti.entities(msh)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, ents)
    Γ = Inti.boundary(Ω)
    Γ_msh = view(msh, Γ)

    #-

    ## The incident field
    d = SVector(cos(θ), sin(θ)) # incident direction
    uᵢ = (x) -> exp(im * k * dot(x, d)) # incident plane wave

    ## Create the quadrature
    Q = Inti.Quadrature(Γ_msh; qorder)
    println("Number of quadrature points: ", length(Q))
    range_sizes[i] = length(Q)
    ## Setup the integral operators
    pde = Inti.Helmholtz(; dim = 2, k)
    ## Right-hand side given by Dirichlet trace of plane wave
    g = map(Q) do q
        # normal derivative of e^{ik*d⃗⋅x}
        x, ν = q.coords, q.normal
        return -uᵢ(x)
    end ## Neumann trace on boundary

    ## Use GMRES to solve the linear system
    ε = 1e-8
    σ = 1e-3
    Lop = Inti.IntegralOperator(K,Q,Q)
    L = Inti.assemble_hmatrix(Lop; rtol = ε)
    Id = sparse((0.5 + 0*im)I,size(L))
    axpy!(1.0,Id,L) # in place sum of L



    y_exact = similar(g)
    y_approx = similar(g)


    H_iprod = HMatrices.ITerm(L, 0.0)

    benchr_approx = @benchmark igmres($L,$g,tol=$σ) 
    benchr_exact = @benchmark InexactGMRES.test_gmres($L,$g,tol=$σ)
    

    results_exact[i] = minimum(benchr_exact).time
    results_approx[i] = minimum(benchr_approx).time

    #y_exact = gmres(L,g;reltol=σ)
    
    y_approx,res_aprox,it = igmres(L,g,tol=σ)

    results_itn[i] = it
    push!(rel_error_sol, norm(L*y_approx - g)/norm(g))
    println("Current % of measurement: ",100*(i/length(range_values)))
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

df = DataFrame("Rel. error" => rel_error_sol, "Speed up" => speed_up, "Iterations"=>results_itn,"Size"=>range_sizes)
CSV.write("Fixtol_cavity_results.csv", df)
