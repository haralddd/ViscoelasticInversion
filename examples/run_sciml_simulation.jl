using Revise
using ViscoelasticInversion
using CUDA

# Choose device
device = CUDA.has_cuda() ? CUDABackend() : CPU()

## --- testing
Nx = 256
Nz = 256
h = 100 # meters
println("$(h*Nx)")
ρ = 2600.0 # kg/m³
λ = 36.9e9 # Pa
μ = 30.65e9 # Pa
model = IsotropicModel(ρ, λ, μ, Nx, Nz, device=device)
preallocated = Preallocated(Nx, Nz, device=device)
# TODO: Look into if FDM and BC can be initialized together (or better initializer for Boundary Conditions)
fdm = Stencil(8, h, device=device)
# bc = PeriodicBC(fdm, Nx, Nz)
bc = NeumannBC(fdm, Nx, Nz)
tc = 1.0
fc = 10.0
source = RickerSource(fc, tc, Nx÷2, Nz÷2; device=device)

p = Parameters(Nx, Nz, model, preallocated, fdm, bc, source)
tspan=(0.0,2.0)

sol = try
    timestamp = now()
    printstyled("--- $timestamp: Starting Solve ---\n", bold=true, color=:green)
    prob = make_problem(p, tspan=tspan)

    solve_problem(prob)
catch e
    dumpfile = open("stacktrace.log", "w")
    Base.showerror(dumpfile, e, catch_backtrace())
    println("Error dumped to `stacktrace.log`")

    nothing
end

saved_values = sol.saveval
gr()  # GR backend is more stable

Δt = 0.01
b = 1/ρ
# Create animation of velocity magnitude
anim = @animate for (t, v) in zip(saved_values.t, saved_values.saveval)
    
    heatmap(v[:,:,2],
            title="Velocity Magnitude at t=$(round(t, digits=3))",
            clims=(-b,b)
    )
end


const OUTPUT_FOLDER = "output"
import Base./
/(a::String,b::String) = joinpath(a,b)

# Save animation
gif(anim, OUTPUT_FOLDER / "wave_propagation.gif", fps=10)
println("Animation saved as wave_propagation.gif")
