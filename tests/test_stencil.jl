using ViscoelasticInversion

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