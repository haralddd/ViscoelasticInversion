using Test
using ViscoelasticInversion
using KernelAbstractions

@testset "CPML Tests" begin
    
    @testset "CPMLConfig construction" begin
        # Test with explicit d0
        config = CPMLConfig(thickness=10, d0=100.0)
        @test config.thickness == 10
        @test config.d0 == 100.0
        @test config.left == true
        @test config.right == true
        @test config.top == false  # Free surface default
        @test config.bottom == true
        
        # Test with automatic d0 calculation
        config_auto = CPMLConfig(
            thickness=10, 
            vmax=3000.0, 
            dx=10.0,
            reflection_coef=1e-5
        )
        @test config_auto.d0 > 0
        @test config_auto.thickness == 10
    end
    
    @testset "CPMLCoefficients construction" begin
        config = CPMLConfig(thickness=5, d0=100.0, alpha_max=π, kappa_max=1.5)
        Nx, Nz = 32, 32
        dx, dz = 10.0, 10.0
        dt = 0.001
        
        coeffs = ViscoelasticInversion.CPMLCoefficients(config, Nx, Nz, dx, dz, dt)
        
        # Check arrays have correct size
        @test length(coeffs.d_x) == Nx
        @test length(coeffs.d_z) == Nz
        @test length(coeffs.a_x) == Nx
        @test length(coeffs.b_x) == Nx
        
        # Check damping is non-zero only in PML regions
        d_x_cpu = Array(coeffs.d_x)
        @test all(d_x_cpu[6:Nx-5] .== 0)  # Interior should be zero
        @test any(d_x_cpu[1:5] .> 0)      # Left PML should have damping
        @test any(d_x_cpu[Nx-4:Nx] .> 0)  # Right PML should have damping
    end
    
    @testset "CPMLMemory construction" begin
        Nx, Nz = 32, 32
        mem = ViscoelasticInversion.CPMLMemory(Nx, Nz)
        
        @test size(mem.ψ_vx_x) == (Nx, Nz)
        @test size(mem.ψ_sxx_x) == (Nx, Nz)
        @test all(mem.ψ_vx_x .== 0)
    end
    
    @testset "CPMLBC construction" begin
        config = CPMLConfig(thickness=5, d0=100.0)
        Nx, Nz = 32, 32
        dx, dz = 10.0, 10.0
        dt = 0.001
        
        cpml_bc = CPMLBC(config, Nx, Nz, dx, dz, dt)
        
        @test cpml_bc.config === config
        @test cpml_bc.coeffs isa ViscoelasticInversion.CPMLCoefficients
        @test cpml_bc.memory isa ViscoelasticInversion.CPMLMemory
    end
    
    @testset "FreeSurfaceBC construction" begin
        fs = FreeSurfaceBC()
        @test fs.depth == 1
        
        fs2 = FreeSurfaceBC(2)
        @test fs2.depth == 2
    end
    
    @testset "Parameters with CPML" begin
        cpml_config = CPMLConfig(thickness=5, d0=100.0)
        
        params = Parameters(
            Nx=32, Nz=32,
            dx=10.0, dz=10.0,
            dt=0.001,
            cpml=cpml_config,
            free_surface=true
        )
        
        @test params.cpml !== nothing
        @test params.cpml isa CPMLBC
        @test params.free_surface !== nothing
        @test params.free_surface isa FreeSurfaceBC
    end
    
    @testset "Problem creation with CPML" begin
        cpml_config = CPMLConfig(thickness=5, d0=100.0)
        
        params = Parameters(
            Nx=32, Nz=32,
            dx=10.0, dz=10.0,
            dt=0.001,
            cpml=cpml_config,
            bc=:neumann
        )
        
        problem = make_problem(params, tspan=(0.0, 0.1))
        @test problem !== nothing
    end
    
    @testset "Short simulation with CPML" begin
        cpml_config = CPMLConfig(
            thickness=5, 
            d0=50.0,
            alpha_max=π,
            kappa_max=1.0
        )
        
        params = Parameters(
            Nx=32, Nz=32,
            dx=10.0, dz=10.0,
            dt=0.0005,
            cpml=cpml_config,
            bc=:neumann,
            source=RickerSource(20.0, 0.05, 16, 16)
        )
        
        problem = make_problem(params, tspan=(0.0, 0.02))
        
        # Just test that it runs without error
        using OrdinaryDiffEq
        sol = solve(problem, Tsit5(), saveat=0.01)
        
        @test sol.retcode == :Success || sol.retcode == ReturnCode.Success
        @test length(sol.t) > 1
    end
end
