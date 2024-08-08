
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
function my_arnoldi!(Q, H, A::HMatrices.ITerm, current_it)
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
    rel_to_eps(res,tol)

Converts the residue from iteration k-1 and overall desired tolerance into and eps we'll use to approximate the original problem's matrix A.

"""
function rel_to_eps(res::Float64, tol::Float64)
    return min((tol/min(res,1)),1)
end

#next one is a test, mainly to test different IGmres'es versions and iteration modification

function rel_to_eps(bound_factor::Float64,res::Float64, tol::Float64)
    return min(bound_factor*(tol/min(res,1)),1)
end

"""
    igmres_tolstudy(A,b,...)

Modified version of the Inexact GMRES algorithm to study the residues and different limits used and fix the product tolerances.
All formulas studied come from DOI: 10.1137/S1064827502406415
"""
function igmres_tolstudy(A, b;maxiter=size(A, 2), restart=min(length(b), size(A,2)), see_r=false, tol=sqrt(eps()))
    #choose type to create vectors and matrices
    TA = eltype(A)
    Tb = eltype(b)
    T = promote_type(TA, Tb)

    x = zeros(T, size(b))#will hold answer
    #residuals = zeros(real(T), maxiter) # will hold residuals
    residuals_true = Vector{Float64}()
    residuals_tilde = Vector{Float64}()
    residuals_tilde_true = Vector{Float64}()
    it = 0
    bheta = norm(b)
    m = restart
    res = bheta
    push!(residuals_true,res)
    push!(residuals_tilde_true,res)
    push!(residuals_tilde,res)
    current_perror = Float64
    A_iterable = HMatrices.ITerm(A,res)

    H_singvalues = Vector{Float64}()
    bound_left4 = Vector{Float64}()
    bound_right4 = Vector{Float64}()
    bound_right4H = Vector{Float64}()

    bound_left5 = Vector{Float64}()
    bound_right5 = Vector{Float64}()

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
            if k==1
                current_perror = rel_to_eps(res,tol)
            else
                current_perror = rel_to_eps(H_singvalues[k-1],res,tol)
            end
                A_iterable.rtol = current_perror
            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q, H, A_iterable, k)#no new vector is created, everything is done directly in H and Q
            #---------------------------#

            ###First bound study, using (4.4) from the article, before the transformation of H into a upper triangular matrix            
            dummy_right = 0
            for n=1:k
                dummy_right+=rel_to_eps(residuals_tilde[n],tol)*abs(x[n])
            end
            push!(bound_right4,dummy_right)

            ###We now store the smallest singular value of H before its transformation into a triangular matrix
            matrix_H = zeros(ComplexF32,k+1,k)
            for  i=1:k
                matrix_H[1:(i+1),i] = H[i]
            end
            _,vals,_ = LinearAlgebra.svd(matrix_H)
            smalles_svd = vals[length(vals)]
            push!(H_singvalues,smalles_svd/k)

            dummy_right = 0
            for n=1:k
                dummy_right+=rel_to_eps(H_singvalues[k],residuals_tilde[n],tol)*abs(x[n])
            end
            push!(bound_right4H,dummy_right)


            ###Givens rotation
            #-----------------------------------
            #Rotations on H to make it triangular
            my_rotation!(H, J, e1, k)
            #-------------------------------------

            triangularsquares!(x, H, e1[1:k])

            #Residuals are always stored in the last element of e1
            res = norm(e1[k+1])
            push!(residuals_tilde,res/bheta)

            ##calculating true residual ||Ay - b||
            y=zero(b)
            for i=1:length(Q)
                y+=Q[i]*x[i]
            end
            push!(residuals_true,norm(A*y - b)/norm(b))
            ##
            ##calculating true tile residual ||ro - Vm+1Hmxm||
            y=zero(b)
            for i=1:k
                for j=1:length(H[i])
                    y[j] += H[i][j] * x[i]
                end
            end
            push!(residuals_tilde_true,norm(e1[1:k+1]-y[1:k+1])/norm(b))

            ##
            it += 1
            if see_r
                println("Iteration: ", it, " Current residual: ", res)
            end

            if res < tol #ta zoado esse calculo aqui tb
                y = zero(x)
                for n = 1:k
                    y += Q[n] * x[n]
                end
                bound_left5=map(v->abs(v),y[1:k])
                bound_right5=(1/smalles_svd).*residuals_tilde[1:(length(residuals_tilde)-1)]
                # println("Finished at iteration: ", it + 1, " Final residual: ", res)
                return bound_right4H,bound_right4,bound_left5,bound_right5, H_singvalues,residuals_true,residuals_tilde,residuals_tilde_true, it
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
