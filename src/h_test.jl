using HMatrices, LinearAlgebra, StaticArrays
const Point2D = SVector{2,Float64}

# points on a circle
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