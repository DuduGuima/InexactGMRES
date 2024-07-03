## Load Inti and prepare the environment with weak dependencies

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
#-

#Inexact
using InexactGMRES




## Physical parameters
λ = 0.25 #
k = 2π / λ # wavenumber
θ = π / 4 # angle of incident wave


##Definig custom kernel to make L in a single calculation(and leave it as a hmatrix)
struct Helm_DoubleLayer <: AbstractMatrix{ComplexF64}
    Q::Inti.Quadrature{2,Float64}
    W::Inti.Quadrature{2,Float64}
    k::Float64
end

function Base.getindex(K::Helm_DoubleLayer,i::Int,j::Int)
    r = Inti.coords(K.Q[i]) - Inti.coords(K.W[j])
    d = norm(r)
    ny = Inti.normal(K.W[j])
    k = K.k
    filter = !(d <= Inti.SAME_POINT_TOLERANCE)
    sod_term = im / 4 * hankelh1(0, k * d)
    dod_term = (im * k / 4 / d * hankelh1(1, k * d) .* dot(r, ny))
    return filter*(dod_term - im*k*sod_term)*Inti.weight(K.W[j])
end

Base.size(K::Helm_DoubleLayer) = length(K.Q),length(K.W)
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
S, D = Inti.single_double_layer(;
    pde,
    target = Q,
    source = Q,
    correction = (method = :none,),
    #compression = (method = :hmatrix, tol = 1e-4),
    compression = (method = :none,),
)

## Right-hand side given by Dirichlet trace of plane wave
g = map(Q) do q
    # normal derivative of e^{ik*d⃗⋅x}
    x, ν = q.coords, q.normal
    return -uᵢ(x)
end ## Neumann trace on boundary

## Use GMRES to solve the linear system
L_or = I / 2 + D - im * k * S
#L_or = D - im * k * S
σ = gmres(L, g; restart = 1000, maxiter = 400, abstol = 1e-4, verbose = true)
