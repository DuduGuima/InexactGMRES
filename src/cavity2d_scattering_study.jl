## Load Inti and prepare the environment with weak dependencies
# using CSV
# using DataFrames
using Inti
Inti.stack_weakdeps_env!(; update = false)

#-

## Load the necessary packages
using StaticArrays
using LinearAlgebra
using Meshes
using Gmsh
using GLMakie
using HMatrices
using IterativeSolvers
using LinearMaps
using SpecialFunctions
using SparseArrays
using Plots
using BenchmarkTools
#-

#Inexact
using InexactGMRES
#using InexactGMRES.rel_to_eps



## Physical parameters
λ = 0.25 #
k = 2π / λ # wavenumber
θ = π / 4 # angle of incident wave


##Definig custom kernel to make L in a single calculation(and leave it as a hmatrix)
function helmholtz_custom(target, source, k)
    x,y,ny = Inti.coords(target),Inti.coords(source),Inti.normal(source)
    r = x-y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = im / 4 * hankelh1(0, k * d)
    dod_term = im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny)
    return filter*(dod_term - im*k*sod_term)
end

function single_helmholtz(target,source,k)
    x,y,ny = Inti.coords(target),Inti.coords(source),Inti.normal(source)
    r = x-y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = im / 4 * hankelh1(0, k * d)
    return filter*sod_term
end

function double_helmholtz(target,source,k)
    x,y,ny = Inti.coords(target),Inti.coords(source),Inti.normal(source)
    r = x-y
    d = norm(r)
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    dod_term = im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny)
    return filter*dod_term
end
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

## Setup the integral operators
pde = Inti.Helmholtz(; dim = 2, k)
# S, D = Inti.single_double_layer(;
#     pde,
#     target = Q,
#     source = Q,
#     correction = (method = :none,),
#     #compression = (method = :hmatrix, tol = 1e-4),
#     compression = (method = :none,),
# )
## Right-hand side given by Dirichlet trace of plane wave
g = map(Q) do q
    # normal derivative of e^{ik*d⃗⋅x}
    x, ν = q.coords, q.normal
    return -uᵢ(x)
end ## Neumann trace on boundary

## Use GMRES to solve the linear system
ε = 1e-8
Lop = Inti.IntegralOperator(K,Q,Q)
L = Inti.assemble_hmatrix(Lop; rtol = ε)
Id = sparse((0.5 + 0*im)I,size(L))
axpy!(1.0,Id,L) # in place sum of L

L_matrix = Lop + 0.5*Matrix(I,size(Lop))
_,vals,_ = svd(L_matrix)
order_term = vals[length(vals)]

# δL = Inti.adaptive_correction(Lop;tol=1e-4,maxdist=5*meshsize)
# axpy!(1.0,δL,L) # in place sum of L

#arrays to store results
range_values= [10.0^(-i) for i in 2:Int(-log10(ε))]


results_exact = zeros(length(range_values))
results_approx = zeros(length(range_values))
results_itn = zeros(length(range_values))
rel_error_sol = Vector{Float64}()



y_approx = similar(g)
σ = 1e-6 # choosing the best tolerance in the test results
bound_left44,bound_right44,bound_left54,bound_right54,H_svalues,residuals,iterations = InexactGMRES.igmres_tolstudy(L,g,tol=σ)


# res_exact = history[:resnorm]


_,vals,_=svd(L_matrix)
smallest_sv =vals[length(vals)]

prodct_res = map(x->InexactGMRES.rel_to_eps(x,σ),residuals[1:(length(residuals)-1)])

theoric_bound_A = ((smallest_sv./iterations)*σ)./residuals[1:(length(residuals)-1)]

theoric_bound_H = (H_svalues*σ)./residuals[1:(length(residuals)-1)]

# df = DataFrame("Product Tolerance"=>prodct_res, "Theoric Bound"=>theoric_bound)
# CSV.write("output_teste.csv",df)

p1 = Plots.scatter(1:iterations,[prodct_res theoric_bound_A theoric_bound_H], title="Product tolerance evolution",label=["Article bound" "Theoric bound with A"  "Theoric bound with H"],legend=:bottomright,yaxis=:log)
Plots.xlabel!(p1,"Iteration number")
Plots.ylabel!(p1,"Relative Error")
Plots.ylims!(p1,10.0^(-10),10.0^(-1))
Plots.yticks!(p1,range_values)
Plots.xticks!(p1,1:4:iterations)

p2 = Plots.scatter(1:iterations,[bound_left44 bound_right44 ], title="Bound evolution",label=["(4.4) left side" "(4.4) right side"],legend=:bottomright,yaxis=:log)
Plots.xlabel!(p2,"Iteration number")
Plots.ylabel!(p2,"Bounds")
Plots.ylims!(p2,10.0^(-10),10.0^(-1))
Plots.yticks!(p2,range_values)
Plots.xticks!(p2,1:4:iterations)

p3 = Plots.scatter(1:iterations,[bound_left54 bound_right54 ], title="Bound evolution",label=["(5.4) left side" "(5.4) right side"],legend=true,yaxis=:log)
Plots.xlabel!(p3,"Iteration number")
Plots.ylabel!(p3,"Bounds")
# Plots.ylims!(p3,10.0^(-10),10.0^(-1))
# Plots.yticks!(p3,range_values)
Plots.xticks!(p3,1:4:iterations)


Plots.plot(p2,p3,layout=(2,1))