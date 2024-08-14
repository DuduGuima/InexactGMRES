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
df_cavity = CSV.read("Cavity_results.csv",DataFrame)
df_laplace_fixtol = CSV.read("Fixtol_laplace_results.csv",DataFrame)
df_helmholtz_fixtol = CSV.read("Fixtol_helmholtz_results.csv",DataFrame)
df_cavity_fixtol = CSV.read("Fixtol_cavity_results.csv",DataFrame)

range_sizes = map(x->Int(x),df_helmholtz_fixtol[:,3])

range_sizestring = map(x->string(x),range_sizes)

range_sizescavity = map(x->Int(x),df_cavity_fixtol[:,4])

range_sizecavstring = map(x->string(x),range_sizescavity)

#p1 = Plots.scatter(range_values,[range_values df_laplace[:,1] df_helmholtz[:,1] df_cavity[:,1]], label=["Identity" "Laplace" "Helmholtz Circle" "Cavity"],title="Relative error",yaxis=:log,xaxis=:log)
p1 = Plots.scatter(range_values[1:8],[range_values[1:8] df_laplace[1:8,1] df_helmholtz[:,1] ], label=["Identity" "Laplace" "Helmholtz Circle" ],title="Relative error for the inexact product",yaxis=:log,xaxis=:log, markershape=[:circle :rect :diamond],legend=:bottomright)
Plots.xlabel!(p1,"Inexact Product Tolerance ν")
Plots.ylabel!(p1,L"\textrm{Relative\ Error\ } \frac{||b-Ax_{m}||}{||b||}")
Plots.ylims!(p1,10.0^(-10),10.0^(-1))
Plots.yticks!(p1,range_values)
Plots.xticks!(p1,range_values)


#p2 = Plots.plot(range_values,[df_laplace[:,2] df_helmholtz[:,2] df_cavity[:,2]],label=["Laplace" "Helmholtz Circle" "Cavity"],title="Speed up",xaxis=:log)
p2 = Plots.scatter(range_values[2:8],[df_laplace[2:8,2] df_helmholtz[2:8,2] ],label=["Laplace" "Helmholtz Circle" ],title="Matrix-vector product Speed-up",xaxis=:log,markershape=[:circle :rect],legend=:bottomright)

Plots.xlabel!(p2,"Inexact Product Tolerance ν")
Plots.ylabel!(p2,L"\textrm{Speed\ up\ } \frac{t_{exact}}{t_{inexact}}")
Plots.xticks!(p2,range_values)
Plots.yticks!(p2,LinRange(1,1.6,7))

p3 = Plots.scatter(range_values[2:8],[range_values[2:8] df_cavity[2:8,1]],label=["Identity" "Inexact GMRES residuals"],title="Residual evolution for Inexact GMRES",xaxis=:log,markershape=[:circle :rect],yaxis=:log,legend=:bottomright)

Plots.xlabel!(p3,"Overall tolerance of the algorithm ε")
Plots.ylabel!(p3,L"\textrm{Relative\ residual\ } \frac{||g - Ax||}{||b||}")
Plots.xticks!(p3,range_values)
#Plots.yticks!(p3,LinRange(1,1.6,7))

p4 = Plots.scatter(range_values[2:8],[df_cavity[2:8,2]],title="Speed-up for fixed-size Inexact GMRES",xaxis=:log,markershape=[:circle], legend=:false)

Plots.xlabel!(p4,"Overall tolerance of the algorithm ε")
Plots.ylabel!(p4,L"\textrm{Speed\ up\ } \frac{t_{exact}}{t_{inexact}}")
Plots.xticks!(p4,range_values)
#Plots.yticks!(p4,LinRange(1,1.6,7))

p5 = Plots.scatter(range_values[2:8],[df_cavity[2:8,3]],title="Iterations for fixed-size Inexact GMRES",xaxis=:log,markershape=[:circle], legend=:false)

Plots.xlabel!(p5,"Overall tolerance of the algorithm ε")
Plots.ylabel!(p5,"Iterations until convergence")
Plots.xticks!(p5,range_values)
#Plots.yticks!(p5,LinRange(1,1.6,7))

p6 = Plots.scatter(range_sizes,[repeat([1e-6],length(range_sizes)) df_laplace_fixtol[:,1] df_helmholtz_fixtol[:,1] ], label=["ν" "Laplace" "Helmholtz Circle" ],title="Speed-up for the inexact product",yaxis=:log,xaxis=:log2, markershape=[:circle :rect :diamond],legend=:bottom,legendcolumns=3)
Plots.xlabel!(p6,"Coefficient matrix size")
Plots.ylabel!(p6,L"\textrm{Relative\ Error\ } \frac{||b-Ax_{m}||}{||b||}")
Plots.ylims!(p6,10.0^(-10),10.0^(-5))
Plots.yticks!(p6,range_values)
Plots.xticks!(p6,range_sizes,range_sizestring)


p7 = Plots.scatter(range_sizes,[df_laplace_fixtol[:,2] df_helmholtz_fixtol[:,1] ], label=["Laplace" "Helmholtz Circle" ],title="Matrix-vector product Speed-up",xaxis=:log2, markershape=[:circle :rect],legend=:topright)
Plots.xlabel!(p7,"Coefficient matrix size")
Plots.ylabel!(p7,L"\textrm{Speed\ up\ }\frac{t_{exact}}{t_{inexact}}")

Plots.yticks!(p7,LinRange(1,1.6,7))
Plots.xticks!(p7,range_sizes,range_sizestring)



p8 = Plots.scatter(range_sizescavity,[repeat([1e-6],length(range_sizescavity)) df_cavity_fixtol[:,1]],label=["ν" "Inexact GMRES residuals"],title="Residual evolution for Inexact GMRES with fixed tolerance",xaxis=:log2,markershape=[:circle :rect],yaxis=:log,legend=:bottomright)

Plots.xlabel!(p8,"Overall tolerance of the algorithm ε")
Plots.ylabel!(p8,L"\textrm{Relative\ residual\ } \frac{||g - Ax||}{||b||}")
Plots.xticks!(p8,range_values)
#Plots.yticks!(p8,LinRange(1,1.6,7))
Plots.ylims!(p8,10.0^(-10),10.0^(-5))
Plots.yticks!(p8,range_values)
Plots.xticks!(p8,range_sizescavity,range_sizecavstring)


p9 = Plots.scatter(range_sizescavity,[ df_cavity_fixtol[:,2]],title="Speed-up for fixed tolerance Inexact GMRES",xaxis=:log2,markershape=[:circle], legend=:false)

Plots.xlabel!(p9,"Overall tolerance of the algorithm ε")
Plots.ylabel!(p9,L"\textrm{Speed\ up\ } \frac{t_{exact}}{t_{inexact}}")
Plots.xticks!(p9,range_values)
#Plots.yticks!(p9,LinRange(1,1.6,7))
Plots.yticks!(p9,LinRange(1,1.6,7))
Plots.xticks!(p9,range_sizescavity,range_sizecavstring)




p10 = Plots.scatter(range_sizescavity,[df_cavity_fixtol[:,3]],title="Iterations for fixed tolerance Inexact GMRES",xaxis=:log2,markershape=[:circle], legend=:false)

Plots.xlabel!(p10,"Overall tolerance of the algorithm ε")
Plots.ylabel!(p10,"Iterations until convergence")
Plots.xticks!(p10,range_sizescavity,range_sizecavstring)
#Plots.yticks!(p10,LinRange(1,1.6,7))