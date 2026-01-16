import KernelAbstractions as KA
using KernelAbstractions
import SciMLBase: isautodifferentiable

# Preferred device storage types in the stencil struct
preferred_float(::CPU) = Float64
preferred_float(::GPU) = Float32
preferred_int(::CPU) = Int
preferred_int(::GPU) = Int

function _stencil(x::AbstractVector{<:Real}, x₀::Real, m::Integer)
    ℓ = 0:length(x)-1
    m in ℓ || throw(ArgumentError("order $m ∉ $ℓ"))
    A = @. (x' - x₀)^ℓ / factorial(ℓ)
    return A \ (ℓ .== m) # vector of weights w
end

struct Stencil
    xgrid
    zgrid

    xcoefs
    zcoefs

    # Define interior bounds
    x0
    z0
    x1
    z1

    Stencil(xgrid, zgrid, xcoefs, zcoefs, x0, z0, x1, z1) = new(xgrid, zgrid, xcoefs, zcoefs, x0, z0, x1, z1)

    function Stencil(xgrid, zgrid, x0, z0, Δx, Δz)
        xcoefs = _stencil(Rational.(xgrid), x0, 1) ./ Δx
        zcoefs = _stencil(Rational.(zgrid), z0, 1) ./ Δz

        x0 = abs(min(minimum(xgrid), 0))
        z0 = abs(min(minimum(zgrid), 0))
        x1 = abs(max(maximum(xgrid), 0))
        z1 = abs(max(maximum(zgrid), 0))

        return Stencil(xgrid, zgrid, xcoefs, zcoefs, x0, z0, x1, z1)
    end

    function Stencil(xorder, zorder, Δx, Δz)
        @assert iseven(xorder) && iseven(zorder) "Called Stencil constructor is only defined for even orders"
        xpad = xorder ÷ 2
        zpad = zorder ÷ 2
        xgrid = filter(!iszero, -xpad:xpad)
        zgrid = filter(!iszero, -zpad:zpad)

        Stencil(xgrid, zgrid, 0, 0, Δx, Δz)
    end

    function Stencil(order, h)
        Stencil(order, order, h, h)
    end
end


function to_device(device, fdm::Stencil)::Stencil
    I = preferred_int(device)
    F = preferred_float(device)
    xgrid = allocate(device, I, size(fdm.xgrid))
    zgrid = allocate(device, I, size(fdm.zgrid))
    xcoefs = allocate(device, F, size(fdm.xcoefs))
    zcoefs = allocate(device, F, size(fdm.zcoefs))
    copyto!(xgrid, Vector{I}(fdm.xgrid))
    copyto!(zgrid, Vector{I}(fdm.zgrid))
    copyto!(xcoefs, Vector{F}(fdm.xcoefs))
    copyto!(zcoefs, Vector{F}(fdm.zcoefs))
    return Stencil(xgrid,zgrid,xcoefs,zcoefs,fdm.x0,fdm.z0,fdm.x1,fdm.z1)
end

@kernel unsafe_indices = true function _ddx_kernel!(du, u, grid, coefs, x0, z0)
    I = @index(Global, NTuple)
    m = I[1] + x0
    n = I[2] + z0

    val = zero(eltype(du))
    @inbounds for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        @inbounds val += c * u[m+g, n]
    end
    @inbounds du[m, n] = val
end

@kernel unsafe_indices = true function _ddz_kernel!(du, u, grid, coefs, x0, z0)
    I = @index(Global, NTuple)
    m = I[1] + x0
    n = I[2] + z0

    val = zero(eltype(du))
    @inbounds for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        @inbounds val += c * u[m, n+g]
    end
    @inbounds du[m, n] = val
end

function ddx!(du, u, fdm::Stencil)
    device = get_backend(du)
    kernel! = _ddx_kernel!(device, 64)
    Nx, Nz = size(du)
    ndrange = (Nx - fdm.x0 - fdm.x1, Nz - fdm.z0 - fdm.z1)
    kernel!(du, u, fdm.xgrid, fdm.xcoefs, fdm.x0, fdm.z0; ndrange=ndrange)
    return nothing
end
function ddz!(du, u, fdm::Stencil)
    device = get_backend(du)
    kernel! = _ddz_kernel!(device, 64)
    Nx, Nz = size(du)
    ndrange = (Nx - fdm.x0 - fdm.x1, Nz - fdm.z0 - fdm.z1)
    kernel!(du, u, fdm.zgrid, fdm.zcoefs, fdm.x0, fdm.z0; ndrange=ndrange)
    return nothing
end

function ddx_synced!(du, u, fdm::Stencil)
    device = get_backend(du)
    ddx!(du, u, fdm)
    KA.synchronize(device)
    return nothing
end
function ddz_synced!(du, u, fdm::Stencil)
    device = get_backend(du)
    ddz!(du, u, fdm)
    KA.synchronize(device)
    return nothing
end

function test_timestep!(du, u, fdm::Stencil)
    ddx_synced!(du, u, fdm)
    u = u + du
    return nothing
end



if abspath(PROGRAM_FILE) == @__FILE__
    # -- Timing, benchmarks and tests --
    using Random
    using CUDA
    using Statistics
    using Enzyme
    using ProgressMeter

    # Testparams
    M = 1024
    N = 1024
    iters = 1000

    u_cpu = zeros(M, N)
    u_gpu = cu(u_cpu)

    # Test automatic differentiation
    println("Testing AD using Enzyme")
    du_cpu_f = similar(u_cpu)
    du_gpu_f = similar(u_gpu)

    du_cpu_r = similar(u_cpu)
    du_gpu_r = similar(u_gpu)

    fdm_cpu = Stencil(8,1)
    fdm_gpu = to_device(get_backend(u_gpu), fdm_cpu)


    # https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/#Vector-Jacobian-Product-(VJP)-Choices
    loss(u,du,fdm) = sum()
    # For in-place functions, use autodiff_thunk
    println("Testing reverse AD for in-place function...")


    
    
    println("Gradient computation completed!")
    display(du_cpu_r)

    # Timing
    function time_fdm_kernel(u, iters, direction=:x, label="")
        du = similar(u)

        device = get_backend(u)
        fdm = Stencil(8, 1.0)
        fdm = to_device(device, fdm)
        if direction == :x
            func = (du, u) -> ddx_synced!(du, u, fdm)
        else
            func = (du, u) -> ddz_synced!(du, u, fdm)
        end

        timings = zeros(iters)
        p = Progress(iters; dt=0.1)
        for i in 1:iters
            randn!(u)
            timings[i] = (@timed func(du, u)).time
            next!(p)
        end

        _median = median(timings) * 1e3
        _std = sqrt(var(timings)) * 1e3

        println("$label: median $_median ms (σ = $_std)")
        return _median, _std
    end

    println("Timing kernels ... ")

    cxt, std_cxt = time_fdm_kernel(u_cpu, iters, :x, "CPU dx")
    czt, std_czt = time_fdm_kernel(u_cpu, iters, :z, "CPU dz")
    gxt, std_gxt = time_fdm_kernel(u_gpu, iters, :x, "GPU dx")
    gzt, std_gzt = time_fdm_kernel(u_gpu, iters, :z, "GPU dz")
    
    println("CPU/GPU x: $(cxt/gxt)")
    println("CPU/GPU z: $(czt/gzt)")
end