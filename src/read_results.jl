using CSV
using Plots
using DataFrames

ε = 1e-8
range_values= [10.0^(-i) for i in 2:Int(-log10(ε))]

df_laplace = CSV.read("Laplace_results.csv",DataFrame)
df_helmholtz = CSV.read("Helmholtz_gmres_results.csv",DataFrame)
df_cavity = CSV.read("Cavity_results.csv",DataFrame)

p1 = Plots.scatter(range_values,[range_values df_laplace[:,1] df_helmholtz[:,1] df_cavity[:,1]], label=["Identity" "Laplace" "Helmholtz Circle" "Cavity"],title="Relative error",yaxis=:log,xaxis=:log)
Plots.xlabel!(p1,"Overall Tolerance σ")
Plots.ylabel!(p1,"Relative Error")
Plots.ylims!(p1,10.0^(-10),10.0^(-1))
Plots.yticks!(p1,range_values)
Plots.xticks!(p1,range_values)


p2 = Plots.plot(range_values,[df_laplace[:,2] df_helmholtz[:,2] df_cavity[:,2]],label=["Laplace" "Helmholtz Circle" "Cavity"],title="Speed up",xaxis=:log)
Plots.xlabel!(p2,"Overall Tolerance σ")
Plots.ylabel!(p2,"Speed up")
Plots.xticks!(p2,range_values)

Plots.plot(p1,p2,layout=(2,1))