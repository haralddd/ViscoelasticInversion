using ViscoelasticInversion

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
source = RickerSource(fc, tc, Nx÷2, Nz÷2)

p = WaveParams(model, preallocated, fdm, bc, source)
tspan=(0.0,2.0)

T = preferred_float(device)

s0 = KA.zeros(device, T, Nx, Nz, 3)
v0 = KA.zeros(device, T, Nx, Nz, 2)

ds = similar(s0)
stress_eq!(ds, s0, v0, p, 1.0)

# Save output velocity to CPU using saving callback
function save_func(u,t,integrator)
    return Array(u.x[2])
end
saved_values = SavedValues(Float64, Array{Float32,3})
cb = SavingCallback(save_func, saved_values, saveat=0.0:0.01:2.0)

try
    timestamp = now()
    printstyled("--- $timestamp: Starting Solve ---\n", bold=true, color=:green)
    prob = DynamicalODEProblem(
        stress_eq!, velocity_eq!, 
        s0, v0, 
        tspan, p)
    solve(prob, callback=cb, progress=true, tstops=[tc], save_on=false, save_start=false, save_end=false)
catch e
    println("Full error:")
    dumpfile = open("stacktrace.log", "w")
    Base.showerror(dumpfile, e, catch_backtrace())
    println("Dumped to `stacktrace.log`")
    # println("\nStack trace:")
    # for (i, frame) in enumerate(stacktrace(catch_backtrace()))
    #     println("$i. $frame")
    # end
    nothing
end

saved_values.saveval
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
