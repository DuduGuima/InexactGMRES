using CSV
using Plots
using DataFrames
using LaTeXStrings
ε = 1e-12
range_values=Vector{Float64}()
push!(range_values,Inf)

for i=2:Int(-log10(ε))

push!(range_values,10.0^(-i) )
end

df_laplace = CSV.read("Laplace_results.csv",DataFrame)
df_helmholtz = CSV.read("Helmholtz_product_results.csv",DataFrame)
#df_cavity = CSV.read("Cavity_results.csv",DataFrame)

#p1 = Plots.scatter(range_values,[range_values df_laplace[:,1] df_helmholtz[:,1] df_cavity[:,1]], label=["Identity" "Laplace" "Helmholtz Circle" "Cavity"],title="Relative error",yaxis=:log,xaxis=:log)
p1 = Plots.scatter(range_values[1:8],[range_values[1:8] df_laplace[1:8,1] df_helmholtz[:,1] ], label=["Identity" "Laplace" "Helmholtz Circle" ],title="Relative error",yaxis=:log,xaxis=:log, markershape=[:circle :rect :diamond])
Plots.xlabel!(p1,"Overall Tolerance ε")
Plots.ylabel!(p1,L"\textrm{Relative\ Error\ } \frac{||b-Ax_{m}||}{||b||}")
Plots.ylims!(p1,10.0^(-10),10.0^(-1))
Plots.yticks!(p1,range_values)
Plots.xticks!(p1,range_values)


#p2 = Plots.plot(range_values,[df_laplace[:,2] df_helmholtz[:,2] df_cavity[:,2]],label=["Laplace" "Helmholtz Circle" "Cavity"],title="Speed up",xaxis=:log)
p2 = Plots.scatter(range_values[2:8],[df_laplace[2:8,2] df_helmholtz[2:8,2] ],label=["Laplace" "Helmholtz Circle" ],title="Matrix-vector product peed up",xaxis=:log,markershape=[:circle :rect],legend=:bottomright)

Plots.xlabel!(p2,"Overall Tolerance ε")
Plots.ylabel!(p2,L"\textrm{Speed\ up\ } \frac{t_{exact}}{t_{inexact}}")
Plots.xticks!(p2,range_values)
Plots.yticks!(p2,LinRange(1,1.6,7))