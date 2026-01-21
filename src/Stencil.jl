import KernelAbstractions as KA
using KernelAbstractions
include("utils.jl")

function _stencil(x::AbstractVector{<:Real}, x₀::Real, m::Integer)
    ℓ = 0:length(x)-1
    m in ℓ || throw(ArgumentError("order $m ∉ $ℓ"))
    A = @. (x' - x₀)^ℓ / factorial(ℓ)
    return A \ (ℓ .== m) # vector of weights w
end

"""
    Stencil(order, h; device=CPU())
    Stencil(xorder, zorder, Δx, Δz; device=CPU())
    Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())

High-order finite difference stencil for computing spatial derivatives.

# Constructors
- `Stencil(order, h; device=CPU())`: Creates isotropic stencil with given order and spacing
- `Stencil(xorder, zorder, Δx, Δz; device=CPU())`: Creates anisotropic stencil with different orders and spacings
- `Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())`: Creates stencil from custom grid points

# Arguments
- `order`: Finite difference order (must be even)
- `xorder`, `zorder`: Orders in x and z directions (must be even)
- `h`, `Δx`, `Δz`: Grid spacing
- `xgrid`, `zgrid`: Grid point offsets
- `x0`, `z0`: Reference point indices
- `device`: Choose device to allocate indices and coefficients to

# Examples
```Julia
# 8th-order isotropic stencil with unit spacing
stencil = Stencil(8, 1.0)

# 8th-order in x, 4th-order in z with different spacings
stencil = Stencil(8, 4, 1.0, 0.5)

# Custom stencil from grid points
xgrid = [-2, -1, 1, 2]
zgrid = [-2, -1, 1, 2]
stencil = Stencil(xgrid, zgrid, 0, 0, 1.0, 1.0)
```
"""
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

    function Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())

        I = preferred_int(device)
        F = preferred_float(device)
        _xgrid = allocate(device, I, size(xgrid))
        _zgrid = allocate(device, I, size(zgrid))
        _xcoefs = similar(_xgrid, F)
        _zcoefs = similar(_zgrid, F)

        xcoefs = _stencil(Rational.(xgrid), x0, 1) ./ Δx
        zcoefs = _stencil(Rational.(zgrid), z0, 1) ./ Δz

        copyto!(_xgrid, Vector{I}(xgrid))
        copyto!(_zgrid, Vector{I}(zgrid))
        copyto!(_xcoefs, Vector{F}(xcoefs))
        copyto!(_zcoefs, Vector{F}(zcoefs))

        x0 = I(abs(min(minimum(xgrid), 0)))
        z0 = I(abs(min(minimum(zgrid), 0)))
        x1 = I(abs(max(maximum(xgrid), 0)))
        z1 = I(abs(max(maximum(zgrid), 0)))

        return Stencil(_xgrid, _zgrid, _xcoefs, _zcoefs, x0, z0, x1, z1)
    end

    function Stencil(xorder, zorder, Δx, Δz; device=CPU())
        @assert iseven(xorder) && iseven(zorder) "Called Stencil constructor is only defined for even orders"
        xpad = xorder ÷ 2
        zpad = zorder ÷ 2
        xgrid = filter(!iszero, -xpad:xpad)
        zgrid = filter(!iszero, -zpad:zpad)

        Stencil(xgrid, zgrid, 0, 0, Δx, Δz, device=device)
    end

    function Stencil(order, h; device=CPU())
        Stencil(order, order, h, h, device=device)
    end
end


@kernel inbounds = true unsafe_indices = true function _ddx_kernel_padded!(du, u, grid, coefs, x0, z0)
    I = @index(Global, NTuple)
    nx = I[1] + x0
    nz = I[2] + z0

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        val += c * u[nx+g, nz]
    end
    du[nx, nz] = val
end

@kernel inbounds = true unsafe_indices = true function _ddz_kernel_padded!(du, u, grid, coefs, x0, z0)
    I = @index(Global, NTuple)
    nx = I[1] + x0
    nz = I[2] + z0

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        val += c * u[nx, nz+g]
    end
    du[nx, nz] = val
end

@kernel function _ddx_kernel_periodic_left!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nz = I[2]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nx = I[1] + g
        nx += (nx < 1) ? Nx : 0

        val += c * u[nx, nz]
    end
    du[I] = val
end

@kernel function _ddx_kernel_periodic_right!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nz = I[2]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nx = Nx - I[1] + 1 + g
        nx += (nx > Nx) ? -Nx : 0
        
        val += c * u[nx, nz]
    end
    du[I] = val
end

@kernel function _ddx_kernel_periodic_top!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nx = I[1]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nz = I[2] + g
        nz += (nz < 1) ? Nz : 0

        val += c * u[nx, nz]
    end
    du[I] = val
end

@kernel function _ddx_kernel_periodic_bottom!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nx = I[1]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nz = Nz - I[2] + 1 + g
        nz += (nz > Nz) ? -Nz : 0
        
        val += c * u[nx, nz]
    end
    du[I] = val
end

function ddx!(du, u, fdm::Stencil)
    device = get_backend(du)
    kernel! = _ddx_kernel_padded!(device, 64)
    Nx, Nz = size(du)
    ndrange = (Nx - fdm.x0 - fdm.x1, Nz - fdm.z0 - fdm.z1)
    kernel!(du, u, fdm.xgrid, fdm.xcoefs, fdm.x0, fdm.z0; ndrange=ndrange)
    return nothing
end
function ddz!(du, u, fdm::Stencil)
    device = get_backend(du)
    kernel! = _ddz_kernel_padded!(device, 64)
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



if abspath(PROGRAM_FILE) == @__FILE__
    # -- Timing, benchmarks and tests --
    using Random
    using CUDA
    using Statistics
    using ProgressMeter

    # Test periodic boundary kernels
    function test_periodic_kernels()
        println("Testing Periodic Boundary Kernels")
        
        # Small test grid for verification
        Nx, Nz = 16, 16
        u_test = rand(1.0:10.0, Nx, Nz)
        du_test = similar(u_test)
        du_ref = similar(u_test)
        
        # Create stencil
        fdm = Stencil(4, 1.0)  # 4th order for simpler testing
        
        println("Original u at boundaries:")
        println("Left edge: ", u_test[1:3, 8])
        println("Right edge: ", u_test[Nx-2:Nx, 8])
        println("Top edge: ", u_test[8, 1:3])
        println("Bottom edge: ", u_test[8, Nz-2:Nz])
        
        # Test CPU periodic kernels
        device = CPU()
        
        # Test left boundary kernel
        du_left = zeros(4, Nz)  # Only boundary region
        kernel_left = _ddx_kernel_periodic_left!(device)
        kernel_left(du_left, u_test, fdm.xgrid, fdm.xcoefs, Nx, Nz, ndrange=(4, Nz))
        
        # Test right boundary kernel  
        du_right = zeros(4, Nz)  # Only boundary region
        kernel_right = _ddx_kernel_periodic_right!(device)
        kernel_right(du_right, u_test, fdm.xgrid, fdm.xcoefs, Nx, Nz, ndrange=(4, Nz))
        
        # Test top boundary kernel
        du_top = zeros(Nx, 4)  # Only boundary region
        kernel_top = _ddx_kernel_periodic_top!(device)
        kernel_top(du_top, u_test, fdm.xgrid, fdm.xcoefs, Nx, Nz, ndrange=(Nx, 4))
        
        # Test bottom boundary kernel
        du_bottom = zeros(Nx, 4)  # Only boundary region
        kernel_bottom = _ddx_kernel_periodic_bottom!(device)
        kernel_bottom(du_bottom, u_test, fdm.xgrid, fdm.xcoefs, Nx, Nz, ndrange=(Nx, 4))
        
        println("\nPeriodic kernel results:")
        println("Left boundary result: ", du_left[1:3, 8])
        println("Right boundary result: ", du_right[1:3, 8])
        println("Top boundary result: ", du_top[8, 1:3])
        println("Bottom boundary result: ", du_bottom[8, 1:3])
        
        # Test GPU version if available
        if CUDA.has_cuda()
            println("\nTesting GPU periodic kernels...")
            u_gpu = cu(u_test)
            du_gpu = similar(u_gpu)
            
            gpu_device = CUDABackend()
            
            # Test left boundary on GPU

            fdm = Stencil(4, 1.0, device=gpu_device)
            kernel_left_gpu = _ddx_kernel_periodic_left!(gpu_device)
            kernel_left_gpu(du_gpu, u_gpu, fdm.xgrid, fdm.xcoefs, Nx, Nz, ndrange=(4, Nz))
            
            println("GPU left boundary result: ", Array(du_gpu[1:3, 8]))
            
            # Verify CPU vs GPU match
            cpu_result = du_left[1:3, 8]
            gpu_result = Array(du_gpu[1:3, 8])
            
            if maximum(abs, cpu_result .- gpu_result) < 1e-6
                println("✓ CPU and GPU results match!")
            else
                println("✗ CPU and GPU results differ!")
                println("CPU: ", cpu_result)
                println("GPU: ", gpu_result)
            end
        end
        
        println("\nPeriodic kernel test completed!")
    end
    
    # Run the test
    test_periodic_kernels()

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

    fdm_cpu = Stencil(8,1, device=CPU())

    # Timing
    function time_fdm_kernel(u, iters, direction=:x, label="")
        du = similar(u)

        device = get_backend(u)
        fdm = Stencil(8, 1.0, device=device)
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