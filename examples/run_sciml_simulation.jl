using Revise
using ViscoelasticInversion
using CUDA
using Dates
using Plots

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
    print(read(dumpfile, String))
    close(dumpfile)
    println("Error dumped to `stacktrace.log`")

    nothing
end

max_val = maximum(stack([s[2] for s in sol.saveval]))
argmax_val = argmax(stack([s[2] for s in sol.saveval]))
println("Max value: $max_val at index $argmax_val")

gr()

Δt = 0.01
b = 1/ρ
# Create animation of velocity magnitude
anim = @animate for (t, val) in zip(sol.t, sol.saveval)
    s = val[1]
    v = val[2]

    sxx = @view s[:,:,1]
    szz = @view s[:,:,2]
    sxz = @view s[:,:,3]

    vx = @view v[:,:,1]
    vz = @view v[:,:,2]

    hm_sxx = heatmap(sxx,
            title="sxx",
            clims=(-b,b)
    )
    hm_szz = heatmap(szz,
            title="szz",
            clims=(-b,b)
    )
    hm_sxz = heatmap(sxz,
            title="sxz",
            clims=(-b,b)
    )

    hm_vx = heatmap(vx,
            title="vx",
            clims=(-b,b)
    )
    hm_vz = heatmap(vz,
            title="vz",
            clims=(-b,b)
    )
    plot(hm_sxx, hm_szz, hm_sxz, hm_vx, hm_vz, layout=(2, 3), size=(1200, 800), title="Time: $(round(t, digits=3))s")
end


const OUTPUT_FOLDER = "output"

# Save animation
gif(anim, joinpath(OUTPUT_FOLDER, "wave_propagation.gif"), fps=10)
println("Animation saved as wave_propagation.gif")