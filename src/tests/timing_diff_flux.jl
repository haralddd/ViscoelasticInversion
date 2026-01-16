using Flux, NNlib, FiniteDifferences, CUDA, cuDNN, ProgressMeter

function get_central_coeffs(order::Int, dx::T) where T
    @assert order % 2 == 0 "Order must be even for central difference"

    n_points = order + 1
    fdm = central_fdm(n_points, 1)
    
    coeffs = T.(collect(fdm.coefs) ./ dx)
    return coeffs
end

struct SpatialScheme{T, AT<:AbstractArray{T, 4}}
    order::Int
    dx::T
    dz::T
    cxs::AT
    czs::AT
    
    function SpatialScheme(order::Int, dx::T, dz::T) where T
        cxs_coeffs = get_central_coeffs(order, dx)
        czs_coeffs = get_central_coeffs(order, dz)
        
        n_coeffs = length(cxs_coeffs)
        cxs = reshape(cxs_coeffs, n_coeffs, 1, 1, 1)
        czs = reshape(czs_coeffs, 1, n_coeffs, 1, 1)
        
        new{T, typeof(cxs)}(order, dx, dz, cxs, czs)
    end
    
    function SpatialScheme{T, AT}(order::Int, dx::T, dz::T, cxs::AT, czs::AT) where {T, AT<:AbstractArray{T, 4}}
        new{T, AT}(order, dx, dz, cxs, czs)
    end
end

function ddx(scheme::SpatialScheme, field::AbstractArray)
    return NNlib.conv(field, scheme.cxs)
end

function ddz(scheme::SpatialScheme, field::AbstractArray)
    return NNlib.conv(field, scheme.czs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Statistics, LinearAlgebra
    
    N = 4096
    M = 4096
    order = 8
    iters = 1000
    
    println("Testing CPU performance...")
    println("Julia threads: $(Threads.nthreads())")
    println("BLAS threads: $(BLAS.get_num_threads())")
    BLAS.set_num_threads(Threads.nthreads())
    println("Set BLAS threads: $(BLAS.get_num_threads())")

    cpu_u = randn(Float32, M, N, 1, 1)
    scheme_cpu = SpatialScheme(order, 1.0f0, 1.0f0)
    
    cxts = zeros(iters)
    @showprogress for i in 1:iters
        cxts[i] = @elapsed (_ = ddx(scheme_cpu, cpu_u))
    end
    
    czts = zeros(iters)
    @showprogress for i in 1:iters
        czts[i] = @elapsed (_ = ddz(scheme_cpu, cpu_u))
    end
    
    println("Testing GPU performance...")
    gpu_u = CuArray(cpu_u)
    cxs_gpu = CuArray(scheme_cpu.cxs)
    czs_gpu = CuArray(scheme_cpu.czs)
    scheme_gpu = SpatialScheme{Float32, typeof(cxs_gpu)}(order, 1.0f0, 1.0f0, cxs_gpu, czs_gpu)
    
    _ = ddx(scheme_gpu, gpu_u)
    CUDA.synchronize()
    
    gxts = zeros(iters)
    @showprogress for i in 1:iters
        gxts[i] = @elapsed begin
            _ = ddx(scheme_gpu, gpu_u)
            CUDA.synchronize()
        end
    end
    
    gzts = zeros(iters)
    @showprogress for i in 1:iters
        gzts[i] = @elapsed begin
            _ = ddz(scheme_gpu, gpu_u)
            CUDA.synchronize()
        end
    end
    
    cxt = median(cxts)
    czt = median(czts)
    gxt = median(gxts)
    gzt = median(gzts)
    
    println("CPU x time: $(cxt * 1e3) ms")
    println("CPU z time: $(czt * 1e3) ms")
    println("GPU x time: $(gxt * 1e3) ms")
    println("GPU z time: $(gzt * 1e3) ms")
    println("CPU/GPU time x: $(cxt / gxt)")
    println("CPU/GPU time z: $(czt / gzt)")
end
