using Test
using Random
using LinearAlgebra
using InexactGMRES

@test true==true

@testset "check convergence" begin
    m=200
    n_iterations = 18
    A = Matrix((2.0+0im)*I,m,m) + 0.5*randn(m,m)/sqrt(m)
    b = ones(ComplexF64,m,1)

    D = Matrix((1.0+0im)*I ,m,m)

    for i=0:m-1
        D[i+1,i+1] = (-2 +2*sin((i*pi)/(m-1))) + cos((i*pi)/(m-1))im
    end

    A_slow = A + D

    x_fast,residuals_fast,fast_it = igmres(A,b)
    x_slow,residuals_slow,slow_it = igmres(A_slow,b)

    x_ftrue = A\b
    x_strue = A_slow\b

    @test norm(A*x_fast - b) < 1e-10
    @test norm(A_slow*x_slow - b) < 1e-10
    @test fast_it<slow_it

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
