include("Equations.jl")
using CUDA

# Choose device
device = CUDA.has_cuda() ? CUDABackend() : CPU()

## --- testing
Nx = 512
Nz = 512
ρ = 2600.0 # kg/m³
λ = 36.9e9 # Pa
μ = 30.65e9 # Pa
model = IsotropicModel(ρ, λ, μ, Nx, Nz, device=device)
preallocated = Preallocated(Nx, Nz, device=device)
# TODO: Look into if FDM and BC can be initialized together (or better initializer for Boundary Conditions)
fdm = Stencil(8, 1.0, device=device)
bc = ZeroBC(fdm, Nx, Nz)
source = RickerSource(40, 1.0, 256, 256)

p = construct_params(model, preallocated, fdm, bc, source)

T = preferred_float(device)
s0 = KA.zeros(device, T, Nx, Nz, 3)
v0 = KA.zeros(device, T, Nx, Nz, 2)
tspan = (0,10)

dv = similar(v0)
ds = similar(s0)

println("Stress:")
@time stress_eq!(ds, s0, v0, p, 0.0)
@time stress_eq!(ds, s0, v0, p, 0.0)

println("Velocity:")
@time velocity_eq!(dv, s0, v0, p, 1.01)
@time velocity_eq!(dv, s0, v0, p, 1.01)

sol = try
    prob = DynamicalODEProblem(stress_eq!, velocity_eq!, s0, v0, tspan, p)
    solve(prob)
catch e
    println("Full error:")
    Base.showerror(stdout, e, catch_backtrace())
    println("\nStack trace:")
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        println("$i. $frame")
    end
    nothing
end


using Plots
typeof(sol(1).x)
st = Array(sol(1).x[1])
vt = Array(sol(1).x[2])

anim = @animate for i in 1:50
    
    heatmap(sol[:,:,1])
end