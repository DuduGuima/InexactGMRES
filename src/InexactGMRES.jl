module InexactGMRES

using LinearAlgebra
using HMatrices

export igmres


"""
    rel_to_eps(res,tol)

Converts the residue from iteration k-1 and overall desired tolerance into and eps we'll use to approximate the original problem's matrix A.

"""
function rel_to_eps(res::Float64, tol::Float64)
    return min((tol/min(res,1)),1)
end

"""
    triangularsquares!(y,A,b)

Solves the system Ax = b with backwards substitution, assuming A is always upper triangular.
A must be a vector of vectors, where each A[i] is a column vector added through push!(A,v).
"""

function triangularsquares!(y, A, b)
    #y is required to have same length as b
    #A is supposed to be upper triangular, and square
    #Here, A is also an vector of vectors, so the expression need to be adapted
    #b is the RHS, same length as A
    m = length(b)
    #x = zeros(ComplexF64,m)#always in complex values
    for k = m:-1:1
        y[k] = b[k]
        for j = m:-1:k+1
            y[k] = y[k] - A[j][k] * y[j]
        end
        y[k] = y[k] / A[k][k]
    end
end


"""
    my_arnoldi(Q,H,A,current_it)

Calculates the current_it'th iteration of the Arnoldi's Method.
Q and H are assumed as vectors of vectors, where Q[i] stores the i-th orthonormal vector we'll be using as a basis for Km(A,b)
and H[i] stores de i+1 first values of H's i-th column.
"""
function my_arnoldi!(Q, H, A::Union{HMatrices.HTypes,HMatrices.HTriangular}, current_it, p_tol)
    dummy_v = similar(Q[current_it])
    mul!(dummy_v,A,Q[current_it],1,0,p_tol;threads=false)
    push!(Q, dummy_v) # heavy part should be here
    push!(H,zeros(current_it+1))
    for j = 1:current_it
        H[current_it][j] = (Q[j]') * Q[current_it+1]
        #Q[current_it+1] -= H[current_it][j] * Q[j]
        
        #mul!(C,A,B,alpha,bheta) : C <- (A*B)alpha + bhetaC
        mul!(Q[current_it+1],I,Q[j],-H[current_it][j],1)
    end
    H[current_it][current_it+1] = norm(Q[current_it+1])
    
    #Q[current_it+1] /= H[current_it][current_it+1]
    lmul!(1/H[current_it][current_it+1],Q[current_it+1])
end

function my_arnoldi!(Q, H, A::AbstractMatrix, current_it)
    dummy_v = similar(Q[current_it])
    mul!(dummy_v,A,Q[current_it],1,0;threads=false)
    push!(Q, dummy_v) # heavy part should be here
    push!(H,zeros(current_it+1))
    for j = 1:current_it
        H[current_it][j] = (Q[j]') * Q[current_it+1]
        #Q[current_it+1] -= H[current_it][j] * Q[j]
        
        #mul!(C,A,B,alpha,bheta) : C <- (A*B)alpha + bhetaC
        mul!(Q[current_it+1],I,Q[j],-H[current_it][j],1)
    end
    H[current_it][current_it+1] = norm(Q[current_it+1])
    
    #Q[current_it+1] /= H[current_it][current_it+1]
    lmul!(1/H[current_it][current_it+1],Q[current_it+1])
end



"""
    my_rotation!(H,J,rhs,current_it)

Applies Givens's Operator to rotate H and transform it in a upper triangular matrix.
The sequence of operators is assumed to be stored in J.
It also applies the transformations to the Right Hand Side(RHS).
"""
function my_rotation!(H, J, rhs, current_it)
    #given two integers k and k+1, it will return an object G with 4 numbers:(k,k+1,s,c)
    #b=G*a with a column vector a of length k+1 will return a new vector b which has
    #b[k+1] = 0 and b[k] changed by the rotation
    for j = 1:current_it-1
        
        #H[current_it] = J[j] * H[current_it]
        lmul!(J[j],H[current_it])
    end

    J[current_it], = givens(H[current_it][current_it], H[current_it][current_it+1], current_it, current_it + 1)
    
    #H[current_it] = J[current_it] * H[current_it]
    lmul!(J[current_it],H[current_it])


    #rhs[:] = J[current_it] * rhs
    lmul!(J[current_it],rhs)
    
end


"""
    igmres(...)
"""
function igmres(A, b;maxiter=size(A, 2), restart=min(length(b), maxiter), see_r=false, tol=sqrt(eps()))
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
            it += 1
            if see_r
                println("Iteration: ", it, " Current residual: ", res)
            end

            ###Transformation of current residue and overall tolerance in the new error we'll use
            current_perror = rel_to_eps(res,tol)
            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q, H, A, k,current_perror)#no new vector is created, everything is done directly in H and Q
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
            it += 1
            if see_r
                println("Iteration: ", it, " Current residual: ", res)
            end

            current_perror = rel_to_eps(res,tol)
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
