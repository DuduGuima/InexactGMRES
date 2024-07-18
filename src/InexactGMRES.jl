module InexactGMRES

using LinearAlgebra
using HMatrices

export igmres

include("utils.jl")



"""
    igmres(...)
"""
function igmres(A, b;maxiter=size(A, 2), restart=min(length(b), size(A,2)), see_r=false, tol=sqrt(eps()))
    #choose type to create vectors and matrices
    TA = eltype(A)
    Tb = eltype(b)
    T = promote_type(TA, Tb)

    
    x = zeros(T, size(b))#will hold answer
    #residuals = zeros(real(T), maxiter) # will hold residuals
    residuals = Vector{Float64}()
    it = 0
    bheta = norm(b)
    m = restart
    res = bheta
    current_perror = Float64
    A_iterable = HMatrices.ITerm(A,res)
    while it < maxiter
        Q = Vector{Vector{T}}()
        H = Vector{Vector{T}}()
        J = Vector{Any}(undef, m)#

        #resduals =
        e1 = zeros(T, m + 1, 1)
        e1[1] = bheta
        v = b / bheta
        push!(Q, v)
        for k = 1:m
            if it > maxiter
                break
            end


            ###Transformation of current residue and overall tolerance in the new error we'll use
            current_perror = rel_to_eps(res,tol)
            A_iterable.rtol = current_perror
            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q, H, A_iterable, k)#no new vector is created, everything is done directly in H and Q
            #---------------------------#

            ###Givens rotation
            #-----------------------------------
            #Rotations on H to make it triangular
            my_rotation!(H, J, e1, k)
            #-------------------------------------

            triangularsquares!(x, H, e1[1:k])

            #Residuals are always stored in the last element of e1
            res = norm(e1[k+1])
            #residuals[it] = res/bheta
            push!(residuals,res/bheta)

            it += 1
            if see_r
                println("Iteration: ", it, " Current residual: ", res)
            end

            if res < tol #ta zoado esse calculo aqui tb
                y = zero(x)
                for n = 1:k
                    y += Q[n] * x[n]
                end
                # println("Finished at iteration: ", it + 1, " Final residual: ", res)
                return y, residuals, it
            end
        end
    end #main while loop
    # y = zero(x)
    # for n = 1:length(x)
    #     y += Q[n] * x[n]
    # end
            
                
    # println("Maximum iteration reached")
    throw("Maximum iteration reached")
end


###exact implementation, for comparing reasons

function test_gmres(A, b;maxiter=size(A, 2), restart=min(length(b), maxiter), see_r=false, tol=sqrt(eps()))
    #choose type to create vectors and matrices
    TA = eltype(A)
    Tb = eltype(b)
    T = promote_type(TA, Tb)

    
    x = zeros(T, size(b))#will hold answer
    residuals = Vector{Float64}()
    it = 0
    bheta = norm(b)
    m = restart
    res = bheta
    current_perror = Float64
    
    A_iterable = HMatrices.ITerm(A,res)
    while it < maxiter
        Q = Vector{Vector{T}}()
        H = Vector{Vector{T}}()
        J = Vector{Any}(undef, m)#

        #resduals =
        e1 = zeros(T, m + 1, 1)
        e1[1] = bheta
        v = b / bheta
        push!(Q, v)
        for k = 1:m
            if it >= maxiter
                break
            end
            

            current_perror = rel_to_eps(res,tol)
            A_iterable.rtol = current_perror
            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q, H, A, k)
            #---------------------------#

            ###Givens rotation
            #-----------------------------------
            #Rotations on H to make it triangular
            my_rotation!(H, J, e1, k)
            #-------------------------------------

            triangularsquares!(x, H, e1[1:k])

            #Residuals are always stored in the last element of e1
            res = norm(e1[k+1])
            push!(residuals,res/bheta)

            it += 1
            if see_r
                println("Iteration: ", it, " Current residual: ", res)
            end
            
            if res < tol #ta zoado esse calculo aqui tb
                y = zero(x)
                for n = 1:k
                    y += Q[n] * x[n]
                end
                # println("Finished at iteration: ", it + 1, " Final residual: ", res)
                return y, residuals, it
            end
        end
    end #main while loop
    # y = zero(x)
    # for n = 1:length(x)
    #     y += Q[n] * x[n]
    # end
            
                
    # println("Maximum iteration reached")
    throw("Maximum iteration reached")
end



"""
    trefethen_fast(m)

Create the matrix A from Trefethen and Bau's book, formula 35.17, in examples 35.1 and 35.2
"""
function trefethen_fast(m)
    A = Matrix((2.0 + 0im) * I, m, m) + 0.5 * randn(m, m) / sqrt(m)
    D = Matrix((1.0 + 0im) * I, m, m)
    for i = 0:m-1
        D[i+1, i+1] = (-2 + 2 * sin((i * pi) / (m - 1))) + cos((i * pi) / (m - 1))im
    end
    return A
end

function trefethen_slow(m)
    A = trefethen_fast(m)
    D = Matrix((1.0 + 0im) * I, m, m)
    for i = 0:m-1
        D[i+1, i+1] = (-2 + 2 * sin((i * pi) / (m - 1))) + cos((i * pi) / (m - 1))im
    end
    return A + D
end



end # module InexactGMRES
