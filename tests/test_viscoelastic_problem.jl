using ViscoelasticInversion
using Test
using LinearAlgebra
using Statistics

@testset "ViscoelasticProblem Source Integration" begin
    # Test parameters
    Nx, Nz = 32, 32
    dx, dz = 10.0, 10.0
    freq = 25.0
    tc = 0.1
    source_x, source_z = Nx÷2, Nz÷2
    
    # Create model with simple isotropic properties
    ρ = 2000.0  # density
    λ = 1e9    # first Lamé parameter
    μ = 1e9    # second Lamé parameter (shear modulus)
    model = IsotropicModel(ρ, λ, μ, Nx, Nz)
    
    # Create source
    source = RickerSource(freq, tc, source_x, source_z)
    
    # Create parameters
    params = Parameters(
        Nx=Nx, Nz=Nz, dx=dx, dz=dz,
        model=model, source=source,
        bc=:periodic, fd_order_x=4, fd_order_z=4
    )
    
    @testset "Problem Creation" begin
        # Test that problem can be created
        problem = make_problem(params, tspan=(0.0, 0.5))
        
        # Check that problem has the expected structure
        @test hasfield(typeof(problem), :f)  # Should have function field
        @test hasfield(typeof(problem), :u0)  # Should have initial conditions
        @test hasfield(typeof(problem), :tspan)  # Should have time span
        
        # Check initial conditions
        @test all(problem.u0[1] .≈ 0)  # Initial stress should be zero
        @test all(problem.u0[2] .≈ 0)  # Initial velocity should be zero
        
        # Check that parameters contain the source
        @test problem.p.source == source
        @test problem.p.source.freq == freq
        @test problem.p.source.tc == tc
    end
    
    @testset "Source Injection Before Activation" begin
        # Test that before source activation, fields remain zero
        problem = make_problem(params, tspan=(0.0, 0.05))  # Before tc=0.1
        
        # Initialize state
        s0 = zeros(Float32, Nx, Nz, 3)  # stress components
        v0 = zeros(Float32, Nx, Nz, 2)  # velocity components
        du = similar(s0)  # stress derivative
        dv = similar(v0)  # velocity derivative
        
        # Test velocity equation before source activation
        ViscoelasticInversion.velocity_eq!(dv, s0, v0, params, 0.0)  # t=0.0 < tc-tspan
        
        # Velocity should remain zero before source activation
        @test all(Array(dv) .≈ 0)
        
        # Test stress equation (should remain zero regardless of source)
        ViscoelasticInversion.stress_eq!(du, s0, v0, params, 0.0)
        @test all(Array(du) .≈ 0)
    end
    
    @testset "Source Injection During Activation" begin
        # Test that during source activation, velocity changes
        problem = make_problem(params, tspan=(0.0, 0.2))
        
        # Initialize state
        s0 = zeros(Float32, Nx, Nz, 3)  # stress components
        v0 = zeros(Float32, Nx, Nz, 2)  # velocity components
        du = similar(s0)  # stress derivative
        dv = similar(v0)  # velocity derivative
        
        # Test velocity equation during source activation
        ViscoelasticInversion.velocity_eq!(dv, s0, v0, params, tc)  # t=tc (peak activation)
        
        # Convert to array for testing
        dv_array = Array(dv)
        
        # Vertical velocity should have non-zero values at source location
        dvz = @view dv_array[:, :, 2]
        @test dvz[source_x, source_z] != 0
        
        # The value should match the expected Ricker wavelet at peak (scaled by buoyancy)
        expected_source_val = ViscoelasticInversion._ricker(freq, 0.0)  # = 1.0 at tc
        # Note: The actual value is scaled by buoyancy (1/ρ = 1/2000 = 0.0005)
        expected_scaled_val = expected_source_val / 2000.0
        @test abs(dvz[source_x, source_z] - expected_scaled_val) < 1e-6
        
        # Horizontal velocity should remain zero (source only affects vertical)
        dvx = @view dv_array[:, :, 1]
        @test all(dvx .≈ 0)
        
        # Other points should remain zero (point source)
        dvz_copy = copy(dvz)
        dvz_copy[source_x, source_z] = 0
        @test all(dvz_copy .≈ 0)
    end
    
    @testset "Source Injection After Activation" begin
        # Test that after source activation, fields return to zero
        problem = make_problem(params, tspan=(0.0, 0.3))
        
        # Initialize state
        s0 = zeros(Float32, Nx, Nz, 3)  # stress components
        v0 = zeros(Float32, Nx, Nz, 2)  # velocity components
        du = similar(s0)  # stress derivative
        dv = similar(v0)  # velocity derivative
        
        # Test velocity equation after source activation
        t_after = tc + 0.1  # After main source activity
        ViscoelasticInversion.velocity_eq!(dv, s0, v0, params, t_after)
        
        # Velocity should be zero after source activation
        @test all(Array(dv) .≈ 0)
    end
    
    @testset "Stress Response to Velocity Changes" begin
        # Test that stress changes in response to velocity changes induced by source
        problem = make_problem(params, tspan=(0.0, 0.2))
        
        # Initialize with some velocity (simulating source effect)
        s0 = zeros(Float32, Nx, Nz, 3)  # stress components
        v0 = zeros(Float32, Nx, Nz, 2)  # velocity components
        
        # Add velocity at source location (simulating source injection)
        v0[source_x, source_z, 2] = 1.0  # vertical velocity
        
        du = similar(s0)  # stress derivative
        
        # Test stress equation with non-zero velocity
        ViscoelasticInversion.stress_eq!(du, s0, v0, params, tc)
        
        du_array = Array(du)
        
        # Stress should change in response to velocity gradients
        # Due to finite differences, stress changes should propagate from source
        dsxx = @view du_array[:, :, 1]
        dszz = @view du_array[:, :, 2]
        dsxz = @view du_array[:, :, 3]
        
        # At least some stress components should be non-zero
        @test !all(dsxx .≈ 0) || !all(dszz .≈ 0) || !all(dsxz .≈ 0)
    end
    
    @testset "Time Evolution with Source" begin
        # Test time evolution by manually stepping through the equations
        problem = make_problem(params, tspan=(0.0, 0.2))
        
        # Initial state
        s = zeros(Float32, Nx, Nz, 3)  # stress
        v = zeros(Float32, Nx, Nz, 2)  # velocity
        ds = similar(s)  # stress derivative
        dv = similar(v)  # velocity derivative
        
        # Step 1: Before source activation
        ViscoelasticInversion.stress_eq!(ds, s, v, params, 0.0)
        ViscoelasticInversion.velocity_eq!(dv, s, v, params, 0.0)
        @test all(Array(dv) .≈ 0)  # No velocity change before source
        
        # Step 2: During source activation
        ViscoelasticInversion.velocity_eq!(dv, s, v, params, tc)  # At peak
        dv_array = Array(dv)
        @test dv_array[source_x, source_z, 2] != 0  # Vertical velocity at source
        
        # Step 3: Stress responds to velocity changes
        v[source_x, source_z, 2] = dv_array[source_x, source_z, 2]  # Update velocity
        ViscoelasticInversion.stress_eq!(ds, s, v, params, tc)
        ds_array = Array(ds)
        
        # Some stress components should be non-zero due to velocity gradients
        @test !all(ds_array .≈ 0)
        
        # Step 4: After source activation
        ViscoelasticInversion.velocity_eq!(dv, s, v, params, tc + 0.1)
        @test all(Array(dv) .≈ 0)  # No more source injection
    end
    
    @testset "Source Location Verification" begin
        # Test that source injection happens at correct location
        source_x_test, source_z_test = 10, 15
        source_test = RickerSource(freq, tc, source_x_test, source_z_test)
        
        params_test = Parameters(
            Nx=Nx, Nz=Nz, dx=dx, dz=dz,
            model=model, source=source_test,
            bc=:periodic, fd_order_x=4, fd_order_z=4
        )
        
        s0 = zeros(Float32, Nx, Nz, 3)
        v0 = zeros(Float32, Nx, Nz, 2)
        dv = similar(v0)
        
        # Test at peak activation
        ViscoelasticInversion.velocity_eq!(dv, s0, v0, params_test, tc)
        
        dv_array = Array(dv)
        dvz = @view dv_array[:, :, 2]
        
        # Source should be at specified location
        @test dvz[source_x_test, source_z_test] != 0
        expected_scaled_val = 1.0 / 2000.0  # Scaled by buoyancy
        @test dvz[source_x_test, source_z_test] ≈ expected_scaled_val
        
        # Other locations should be zero
        dvz_copy = copy(dvz)
        dvz_copy[source_x_test, source_z_test] = 0
        @test all(dvz_copy .≈ 0)
    end
    
    @testset "Different Source Frequencies" begin
        # Test that different source frequencies produce different behaviors
        freq_low = 10.0
        freq_high = 50.0
        
        source_low = RickerSource(freq_low, tc, source_x, source_z)
        source_high = RickerSource(freq_high, tc, source_x, source_z)
        
        params_low = Parameters(Nx=Nx, Nz=Nz, dx=dx, dz=dz, model=model, source=source_low, bc=:periodic)
        params_high = Parameters(Nx=Nx, Nz=Nz, dx=dx, dz=dz, model=model, source=source_high, bc=:periodic)
        
        s0 = zeros(Float32, Nx, Nz, 3)
        v0 = zeros(Float32, Nx, Nz, 2)
        dv_low = similar(v0)
        dv_high = similar(v0)
        
        # Test at peak activation
        ViscoelasticInversion.velocity_eq!(dv_low, s0, v0, params_low, tc)
        ViscoelasticInversion.velocity_eq!(dv_high, s0, v0, params_high, tc)
        
        # Both should have peak value of 1.0 at tc (scaled by buoyancy)
        expected_scaled_val = 1.0 / 2000.0  # Scaled by buoyancy
        dv_low_array = Array(dv_low)
        dv_high_array = Array(dv_high)
        @test dv_low_array[source_x, source_z, 2] ≈ expected_scaled_val
        @test dv_high_array[source_x, source_z, 2] ≈ expected_scaled_val
        
        # But time spans should be different
        @test source_low.tspan[2] - source_low.tspan[1] > source_high.tspan[2] - source_high.tspan[1]
    end
end

println("All ViscoelasticProblem source integration tests completed!")
