using ViscoelasticInversion
using Test
using LinearAlgebra
using CUDA
using KernelAbstractions: get_backend

# Test the private _ricker function
@testset "Ricker Wavelet Function" begin
    freq = 25.0  # Hz
    
    # Test at time zero (peak)
    t0 = 0.0
    val0 = ViscoelasticInversion._ricker(freq, t0)
    @test val0 ≈ 1.0  # Ricker wavelet has peak value of 1 at t=0
    
    # Test symmetry: ricker(-t) == ricker(t)
    t = 0.01
    val_pos = ViscoelasticInversion._ricker(freq, t)
    val_neg = ViscoelasticInversion._ricker(freq, -t)
    @test val_pos ≈ val_neg
    
    # Test that values decay to near zero at large times
    t_large = 0.1
    val_large = ViscoelasticInversion._ricker(freq, t_large)
    @test abs(val_large) < 0.01
    
    # Test frequency scaling
    freq2 = 50.0
    val_freq2 = ViscoelasticInversion._ricker(freq2, 0.0)
    @test val_freq2 ≈ 1.0  # Should still be 1 at t=0 regardless of frequency
end

@testset "RickerSource Construction" begin
    freq = 25.0
    tc = 0.1
    nx, nz = 10, 15
    
    # Test CPU construction
    source_cpu = RickerSource(freq, tc, nx, nz)
    @test source_cpu.freq == freq
    @test source_cpu.tc == tc
    @test size(source_cpu.grid) == (1, 1)  # Single point source
    @test size(source_cpu.coefs) == (1, 1)  # Single coefficient
    @test source_cpu.grid[1] == (nx, nz)  # Should contain the grid point
    @test source_cpu.coefs[1] == 1.0  # Weight should be 1.0
    @test source_cpu.tspan[1] < tc < source_cpu.tspan[2]
    
    # Test that time span is reasonable (should be around ±1/freq from tc)
    expected_width = 2.0 / freq
    actual_width = source_cpu.tspan[2] - source_cpu.tspan[1]
    @test abs(actual_width - expected_width) < 0.1  # Allow some tolerance
    
    # Test GPU construction if CUDA is available
    if CUDA.has_cuda()
        source_gpu = RickerSource(freq, tc, nx, nz; device=CUDABackend())
        @test source_gpu.freq == freq
        @test source_gpu.tc == tc
        @test size(source_gpu.grid) == (1, 1)  # Single point source
        @test size(source_gpu.coefs) == (1, 1)  # Single coefficient
        @test Array(source_gpu.grid)[1] == (nx, nz)  # Should contain the grid point
        @test get_backend(source_gpu.coefs) == CUDABackend()
        @test Array(source_gpu.coefs)[1] == 1.0  # Weight should be 1.0
    end
end

@testset "RickerSource Call Behavior" begin
    freq = 25.0
    tc = 0.1
    nx, nz = 8, 12
    
    # Test CPU version
    source = RickerSource(freq, tc, nx, nz)
    du = zeros(Float32, nx, nz)
    
    # Test outside time span (should not modify du)
    t_before = source.tspan[1] - 0.01
    du_before = copy(du)
    source(du, t_before)
    @test du == du_before  # Should be unchanged
    
    t_after = source.tspan[2] + 0.01
    du_after = copy(du)
    source(du, t_after)
    @test du == du_after  # Should be unchanged
    
    # Test at peak time (tc)
    du_peak = zeros(Float32, nx, nz)
    source(du_peak, tc)
    
    # The source should inject the peak value (1.0) at exactly one grid point
    expected_val = ViscoelasticInversion._ricker(freq, 0.0)  # = 1.0
    nonzero_indices = findall(!iszero, du_peak)
    @test length(nonzero_indices) == 1  # Only one point should be non-zero
    @test du_peak[nonzero_indices[1]] ≈ expected_val  # That point should have the source value
    @test nonzero_indices[1] == CartesianIndex(nx, nz)  # Should be at the specified grid point
    
    # Test at a time within span but not at peak
    t_offset = tc + 0.01
    du_offset = zeros(Float32, nx, nz)
    source(du_offset, t_offset)
    
    expected_offset = ViscoelasticInversion._ricker(freq, 0.01)
    nonzero_indices_offset = findall(!iszero, du_offset)
    @test length(nonzero_indices_offset) == 1  # Only one point should be non-zero
    @test du_offset[nonzero_indices_offset[1]] ≈ expected_offset  # That point should have the source value
    @test nonzero_indices_offset[1] == CartesianIndex(nx, nz)  # Should be at the specified grid point
    
    # Test that multiple calls accumulate
    du_accumulate = zeros(Float32, nx, nz)
    source(du_accumulate, tc)
    source(du_accumulate, tc)
    nonzero_indices_accum = findall(!iszero, du_accumulate)
    @test length(nonzero_indices_accum) == 1  # Still only one point should be non-zero
    @test du_accumulate[nonzero_indices_accum[1]] ≈ 2.0 * expected_val  # Should be double
    @test nonzero_indices_accum[1] == CartesianIndex(nx, nz)  # Should be at the specified grid point
    
    # Test GPU version if available
    if CUDA.has_cuda()
        source_gpu = RickerSource(freq, tc, nx, nz; device=CUDABackend())
        du_gpu = CUDA.zeros(Float32, nx, nz)
        
        # Test at peak time on GPU
        source_gpu(du_gpu, tc)
        du_gpu_result = Array(du_gpu)
        nonzero_indices_gpu = findall(!iszero, du_gpu_result)
        @test length(nonzero_indices_gpu) == 1  # Only one point should be non-zero
        @test du_gpu_result[nonzero_indices_gpu[1]] ≈ expected_val  # That point should have the source value
        @test nonzero_indices_gpu[1] == CartesianIndex(nx, nz)  # Should be at the specified grid point
        
        # Test accumulation on GPU
        du_gpu2 = CUDA.zeros(Float32, nx, nz)
        source_gpu(du_gpu2, tc)
        source_gpu(du_gpu2, tc)
        du_gpu2_result = Array(du_gpu2)
        nonzero_indices_gpu2 = findall(!iszero, du_gpu2_result)
        @test length(nonzero_indices_gpu2) == 1  # Still only one point should be non-zero
        @test du_gpu2_result[nonzero_indices_gpu2[1]] ≈ 2.0 * expected_val  # Should be double
        @test nonzero_indices_gpu2[1] == CartesianIndex(nx, nz)  # Should be at the specified grid point
    end
end

@testset "RickerSource Edge Cases" begin
    freq = 25.0
    tc = 0.1
    nx, nz = 4, 6  # Small grid for edge case testing
    
    # Test with very small grid
    source_small = RickerSource(freq, tc, 1, 1)
    du_small = zeros(Float32, 1, 1)
    source_small(du_small, tc)
    @test du_small[1] ≈ 1.0
    
    # Test with different threshold values
    source_thresh = RickerSource(freq, tc, nx, nz; thresh=1e-8)
    @test source_thresh.tspan[2] - source_thresh.tspan[1] > 
          source_small.tspan[2] - source_small.tspan[1]  # Lower threshold = wider span
    
    # Test frequency edge cases
    # Very low frequency
    source_low_freq = RickerSource(1.0, tc, nx, nz)
    @test source_low_freq.tspan[2] - source_low_freq.tspan[1] > 2.0  # Should be wide
    
    # Very high frequency
    source_high_freq = RickerSource(100.0, tc, nx, nz)
    @test source_high_freq.tspan[2] - source_high_freq.tspan[1] < 0.1  # Should be narrow
end

@testset "RickerSource Injection Location" begin
    freq = 25.0
    tc = 0.1
    nx, nz = 8, 12
    
    source = RickerSource(freq, tc, nx, nz)
    du = zeros(Float32, nx, nz)
    
    # Test that source injects at the specified grid point (nx, nz)
    source(du, tc)
    expected_val = ViscoelasticInversion._ricker(freq, 0.0)  # = 1.0
    
    # Only the specified point should be non-zero
    nonzero_indices = findall(!iszero, du)
    @test length(nonzero_indices) == 1
    @test nonzero_indices[1] == CartesianIndex(nx, nz)
    @test du[nx, nz] ≈ expected_val
    
    # Test with different grid sizes - should inject at specified point
    source_large = RickerSource(freq, tc, 4, 6)
    du_large = zeros(Float32, 4, 6)
    source_large(du_large, tc)
    nonzero_indices_large = findall(!iszero, du_large)
    @test length(nonzero_indices_large) == 1
    @test nonzero_indices_large[1] == CartesianIndex(4, 6)
    @test du_large[4, 6] ≈ expected_val
    
    # Test that coefficients are properly sized for single point
    @test size(source.grid) == (1, 1)
    @test size(source.coefs) == (1, 1)
    @test source.grid[1] == (nx, nz)
    @test source.coefs[1] == 1.0
end

@testset "Source Type Hierarchy" begin
    freq = 25.0
    tc = 0.1
    nx, nz = 8, 8
    
    source = RickerSource(freq, tc, nx, nz)
    
    # Test that RickerSource is a subtype of AbstractSource
    @test source isa ViscoelasticInversion.AbstractSource
    @test typeof(source) <: ViscoelasticInversion.AbstractSource
end

println("All source tests completed!")
