using Test
using Random
using LinearAlgebra
using InexactGMRES
using IterativeSolvers
using StaticArrays
using HMatrices
using HMatrices: RkMatrix
using LoopVectorization

@test true == true

@testset "check convergence" begin
    m = 1000
    n_iterations = 18
    A = Matrix((2.0 + 0im) * I, m, m) + 0.5 * randn(m, m) / sqrt(m)
    b = ones(ComplexF64, m)

    D = Matrix((1.0 + 0im) * I, m, m)

    for i = 0:m-1
        D[i+1, i+1] = (-2 + 2 * sin((i * pi) / (m - 1))) + cos((i * pi) / (m - 1))im
    end

    A_slow = A + D

    x_fast, residuals_fast, fast_it = igmres(A, b)
    x_slow, residuals_slow, slow_it = igmres(A_slow, b)

    x_ftrue = A \ b
    x_strue = A_slow \ b




    @test norm(A * x_fast - b) < sqrt(eps())
    @test norm(A_slow * x_slow - b) < sqrt(eps())
    @test fast_it < slow_it

end;

@testset "check HMatrix implementation" begin
    Point2D = SVector{2,Float64}
    m = n = 10_000
    X = Y = [Point2D(sin(i*2π/n),cos(i*2π/n)) for i in 0:n-1]
    nothing


    struct LaplaceMatrix <: AbstractMatrix{Float64}
        X::Vector{Point2D}
        Y::Vector{Point2D}
    end
    
    Base.getindex(K::LaplaceMatrix,i::Int,j::Int) = -1/2π*log(norm(K.X[i] - K.Y[j]) + 1e-10)
    Base.size(K::LaplaceMatrix) = length(K.X), length(K.Y)
    
    # create the abstract matrix
    K = LaplaceMatrix(X,Y)

    Xclt = Yclt = ClusterTree(X)
    adm = StrongAdmissibilityStd()
    comp = PartialACA(;atol=1e-6)

    H = assemble_hmatrix(K,Xclt,Yclt;adm,comp,threads=false,distributed=false)
    T = eltype(H)
    b = rand(T,n)

    x_exact = gmres(H,b)
    x_approx, = igmres(H,b) #no mention to tolerance will make algorithm use tol ~ 1e-8

    @test norm(H*x_approx - b) < sqrt(eps())
    @test norm(x_approx - x_exact) / norm(x_exact) < sqrt(eps())
end;

# @testset "check least squares solver" begin
#     for m=10:20
#         A = rand(m,m)
#         b = rand(m)
#         A_t=UpperTriangular(A)

#         my_x = triangularsquares!(A_t,b)
#         true_x = A_t\b

#         res = norm(true_x - my_x)/norm(true_x)

#         @test res < 1e-5
#     end

# end;
