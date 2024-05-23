module InexactGMRES

using LinearAlgebra
using HMatrices

export igmres


function triangularsquares!(y,A,b)
    #y is required to have same length as b
    #A is supposed to be upper triangular, and square
    #b is the RHS, same length as A 
    m = length(b)
    #x = zeros(ComplexF64,m)#always in complex values
    for k=m:-1:1
        y[k] = b[k]
        for j=m:-1:k+1
            y[k] = y[k] - A[k,j]*y[j]
        end
        y[k] = y[k]/A[k,k]
    end
end

function my_arnoldi!(Q,H,A, current_it)
    Q[:,current_it+1] = A*Q[:,current_it] #heavy part should be here
    for j=1:current_it
        H[j,current_it] = (Q[:,j]')*Q[:,current_it+1]
        Q[:,current_it+1]-= H[j,current_it]*Q[:,j]
    end
    H[current_it+1,current_it]=norm(Q[:,current_it+1])
    Q[:,current_it+1]/=H[current_it+1,current_it]
end

function my_rotation!(H,J,rhs,current_it)
    #given two integers k and k+1, it will return an object G with 4 numbers:(k,k+1,s,c)
    #b=G*a with a column vector a of length k+1 will return a new vector b which has
    #b[k+1] = 0 and b[k] changed by the rotation
    for j=1:current_it-1
        H[1:current_it+1,current_it] = J[j] * H[1:current_it+1,current_it]
    end

    J[current_it], = givens(H[current_it,current_it],H[current_it+1,current_it],current_it,current_it+1)

    H[1:current_it+1,current_it] = J[current_it] * H[1:current_it+1,current_it]

    rhs[:]=J[current_it]*rhs
end


function igmres(A,b,maxiter=size(A,2),restart = min(length(b),maxiter),see_r = false,tol=1e-10)

    if eltype(A)==ComplexF64 || eltype(b)==ComplexF64
        T = ComplexF64

    else
        T = eltype(A)
    end
    x = zeros(T,size(b))
    residuals = zeros(Float64,maxiter)

    it=0
    bheta = norm(b)
    m=restart
    res = bheta

    while it<maxiter
        Q=zeros(T,size(A,1),m+1)
        H=zeros(T,m+1,m)

        J = Vector{Any}(undef,m)#

        #resduals = 
        e1=zeros(T,m+1,1)
        e1[1]=bheta
        v = b/bheta
        Q[:,1] = v
        for k=1:m
            if it>= maxiter
                break
            end
            it+=1
            if see_r
                println("Iteration: ",it," Current residual: ",res)
            end
            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q,H,A,k)#no new vector is created, everything is done directly in H and Q
            #---------------------------#

            ###Givens rotation
            #-----------------------------------
            #We use this so that H is upper triangular, which speeds up the
            #least square problem we'll have to solve afterwards, to find y
            my_rotation!(H,J,e1,k)
            #-------------------------------------
            #since H is now upper triangular, we could just make a backward substitution
            triangularsquares!(x,H[1:k,1:k],e1[1:k])

            x=Q[:,1:k]*x[1:k]#-> take out this product?

            #Residuals are always stored in the last element of e1
            res = norm(e1[k+1])
            residuals[it] = res/bheta

            if res < tol #ta zoado esse calculo aqui tb
                println("Finished at iteration: ",it+1," Final residual: ",res)
                return x,residuals,it
            end
        end
    end #main while loop
    println("Maximum iteration reached")
    return x,residuals,it
end 

end # module InexactGMRES
