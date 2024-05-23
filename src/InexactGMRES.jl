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
            v = A*Q[:,k] #main heavy part should be in this matrix-vector produ
            
            for j=1:k
                H[j,k] = (Q[:,j]')*v
                v-= H[j,k]*Q[:,j]
            end
            H[k+1,k]=norm(v)
            Q[:,k+1] = v/H[k+1,k]

            #---------------------------#

            ###Givens rotation
            #-----------------------------------
            #We use this so that H is upper triangular, which speeds up the
            #least square problem we'll have to solve afterwards, to find y
            for j=1:k-1
                H[1:k+1,k] = J[j] * H[1:k+1,k]
            end
            #this givens functions isn't very intuitive
            #given two integers k and k+1, it will return an object with 4 numbers:(k,k+1,s,c)
            #where s and c are the numbers needed to rotate a vector X in a way such that will
            #anulate H[k+1,k] and change H[k,k].
            #it seems the object is pre-programmed to do calculations on vectors where k and k+1
            #make sense, so thats why we have to create e1 as a maxiter + 1 column vector 
            J[k], = givens(H[k,k],H[k+1,k],k,k+1)
            #the loop above only fixes the first k elements, to change the last one:
            H[1:k+1,k] = J[k] * H[1:k+1,k]
            #since now H is rotated in the final linear system, we need to change the RHS too:
            e1=J[k]*e1
            #-------------------------------------
            #y = H[1:k,1:k]\e1[1:k] since H is now upper triangular, we could just make a backward substitution
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
