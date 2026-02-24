
"""
Viscoelastic wave propagation solver with CPML absorbing boundaries.

This module implements:
- Velocity-stress staggered grid formulation
- CFS-CPML absorbing boundary conditions  
- Free surface boundary condition (stress imaging)
- Support for isotropic, VTI, and TTI anisotropy

The equations are solved using SciML's DynamicalODEProblem for the
split velocity-stress system, compatible with reverse-mode AD.
"""

#=============================================================================
# Stress update functions
=============================================================================#

"""
    _update_ds!(ds, prealloc::Preallocated, model::T) where T<:AbstractModel

Update the stress field rate `ds` from velocity gradients using constitutive relation.
"""
function _update_ds!(ds, prealloc::Preallocated, model::T) where T<:AbstractModel
    error("_update_ds! not implemented for $T")
end

"TTI model: Uses the full stiffness matrix including C15, C35"
function _update_ds!(ds, prealloc::Preallocated, model::TTIModel)
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11, C13, C15 = model.C11, model.C13, model.C15
    C33, C35, C55 = model.C33, model.C35, model.C55

    dxvx, dzvx = prealloc.dxvx, prealloc.dzvx
    dxvz, dzvz = prealloc.dxvz, prealloc.dzvz

    @. dsxx = C11 * dxvx + C13 * dzvz + C15 * (dzvx + dxvz)
    @. dszz = C13 * dxvx + C33 * dzvz + C35 * (dzvx + dxvz)
    @. dsxz = C15 * dxvx + C35 * dzvz + C55 * (dzvx + dxvz)
end

"VTI/Isotropic model: Uses C11, C13, C33, C55 only"
function _update_ds!(ds, prealloc::Preallocated, model::Union{IsotropicModel,VTIModel})
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11, C13, C33, C55 = model.C11, model.C13, model.C33, model.C55
    dxvx, dzvx = prealloc.dxvx, prealloc.dzvx
    dxvz, dzvz = prealloc.dxvz, prealloc.dzvz

    @. dsxx = C11 * dxvx + C13 * dzvz
    @. dszz = C13 * dxvx + C33 * dzvz
    @. dsxz = C55 * (dzvx + dxvz)
end

"Visco TTI model: Uses the full stiffness matrix including C15, C35"
function _update_ds!(ds, prealloc::Preallocated, model::ViscoTTIModel)
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11, C13, C15 = model.C11, model.C13, model.C15
    C33, C35, C55 = model.C33, model.C35, model.C55

    dxvx, dzvx = prealloc.dxvx, prealloc.dzvx
    dxvz, dzvz = prealloc.dxvz, prealloc.dzvz

    # all auxiliary fields
    N = size(ds, 3) - 3
    for n in 1:N
        dot_eps = dxvx
        N_inv = 1.0 / N
        τn_inv = 1.0 / τn
        dM11 = - 0.5 * N_inv * τn_inv * dot_eps - τn_inv * M11
    end

    @. dsxx = C11 * dxvx + C13 * dzvz + C15 * (dzvx + dxvz)
    @. dszz = C13 * dxvx + C33 * dzvz + C35 * (dzvx + dxvz)
    @. dsxz = C15 * dxvx + C35 * dzvz + C55 * (dzvx + dxvz)
end

"Visco VTI/Isotropic model: Uses C11, C13, C33, C55 only"
function _update_ds!(ds, prealloc::Preallocated, model::Union{ViscoIsotropicModel,ViscoVTIModel})
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11, C13, C33, C55 = model.C11, model.C13, model.C33, model.C55
    dxvx, dzvx = prealloc.dxvx, prealloc.dzvx
    dxvz, dzvz = prealloc.dxvz, prealloc.dzvz

    @. dsxx = C11 * dxvx + C13 * dzvz
    @. dszz = C13 * dxvx + C33 * dzvz
    @. dsxz = C55 * (dzvx + dxvz)
end

#=============================================================================
# Main ODE right-hand-side functions
=============================================================================#

"""
    stress_eq!(ds, s, v, p::Parameters, t)

Compute stress rate from velocity field.
Applies CPML corrections if configured.
"""
function stress_eq!(ds, s, v, p::Parameters, t)
    vx = @view v[:, :, 1]
    vz = @view v[:, :, 2]

    model = p.model
    prealloc = p.prealloc
    fdm⁺ = p.fdm_plus
    fdm⁻ = p.fdm_minus
    bc! = p.bc
    cpml = p.cpml

    dxvx, dzvx = prealloc.dxvx, prealloc.dzvx
    dxvz, dzvz = prealloc.dxvz, prealloc.dzvz

    device = get_backend(dxvx)
    
    # Compute spatial derivatives
    ddx!(dxvx, vx, fdm⁺)
    ddz!(dzvx, vx, fdm⁺)
    ddx!(dxvz, vz, fdm⁺)
    ddz!(dzvz, vz, fdm⁺)
    KA.synchronize(device)

    # Apply boundary conditions to derivatives
    bc!(dxvx)
    bc!(dzvx)
    bc!(dxvz)
    bc!(dzvz)
    KA.synchronize(device)

    # Apply CPML if configured
    if cpml !== nothing
        _apply_cpml_velocity_derivatives!(prealloc, cpml)
        KA.synchronize(device)
    end

    # Update stress from strain rate
    _update_ds!(ds, prealloc, model)

    return nothing
end

"""
    velocity_eq!(dv, s, v, p::Parameters, t)

Compute velocity rate from stress divergence.
Applies CPML corrections and free surface BC if configured.
"""
function velocity_eq!(dv, s, v, p::Parameters, t)
    model = p.model
    prealloc = p.prealloc
    fdm⁺ = p.fdm_plus
    fdm⁻ = p.fdm_minus
    bc! = p.bc
    cpml = p.cpml
    free_surface = p.free_surface
    source! = p.source

    dx_sxx = prealloc.dx_sxx
    dx_szx = prealloc.dx_szx
    dz_sxz = prealloc.dz_sxz
    dz_szz = prealloc.dz_szz

    sxx = @view s[:, :, 1]
    szz = @view s[:, :, 2]
    sxz = @view s[:, :, 3]

    device = get_backend(dx_sxx)
    
    # Apply free surface BC before computing derivatives
    if free_surface !== nothing
        apply_free_surface!(s, free_surface)
        KA.synchronize(device)
    end

    # Compute stress derivatives
    ddx!(dx_sxx, sxx, fdm⁻)
    ddx!(dx_szx, sxz, fdm⁻)
    ddz!(dz_sxz, sxz, fdm⁻)
    ddz!(dz_szz, szz, fdm⁻)
    KA.synchronize(device)

    # Apply boundary conditions
    bc!(dx_sxx)
    bc!(dx_szx)
    bc!(dz_sxz)
    bc!(dz_szz)
    KA.synchronize(device)

    # Apply CPML if configured
    if cpml !== nothing
        _apply_cpml_stress_derivatives!(prealloc, cpml)
        KA.synchronize(device)
    end

    # Compute velocity rate from momentum balance
    dvx = @view dv[:, :, 1]
    dvz = @view dv[:, :, 2]
    b = model.b  # buoyancy = 1/ρ
    
    @. dvx = dx_sxx + dz_sxz
    @. dvz = dx_szx + dz_szz
    
    # Add source (before buoyancy scaling, so source is also scaled)
    source!(dvz, t)
    KA.synchronize(device)
    
    # Apply buoyancy scaling
    dvx .*= b
    dvz .*= b

    return nothing
end

#=============================================================================
# CPML helper functions
=============================================================================#

"""
Apply CPML corrections to velocity derivatives (for stress equation).
"""
function _apply_cpml_velocity_derivatives!(prealloc::Preallocated, cpml::CPMLBC)
    coeffs = cpml.coeffs
    mem = cpml.memory
    
    # Apply CPML to each velocity derivative
    # Note: For now we apply simple κ scaling; full memory update is in separate ODE
    _cpml_scale_x!(prealloc.dxvx, coeffs.kappa_x, mem.ψ_vx_x)
    _cpml_scale_z!(prealloc.dzvx, coeffs.kappa_z, mem.ψ_vx_z)
    _cpml_scale_x!(prealloc.dxvz, coeffs.kappa_x, mem.ψ_vz_x)
    _cpml_scale_z!(prealloc.dzvz, coeffs.kappa_z, mem.ψ_vz_z)
end

"""
Apply CPML corrections to stress derivatives (for velocity equation).
"""
function _apply_cpml_stress_derivatives!(prealloc::Preallocated, cpml::CPMLBC)
    coeffs = cpml.coeffs
    mem = cpml.memory
    
    _cpml_scale_x!(prealloc.dx_sxx, coeffs.kappa_x, mem.ψ_sxx_x)
    _cpml_scale_x!(prealloc.dx_szx, coeffs.kappa_x, mem.ψ_sxz_x)
    _cpml_scale_z!(prealloc.dz_sxz, coeffs.kappa_z, mem.ψ_sxz_z)
    _cpml_scale_z!(prealloc.dz_szz, coeffs.kappa_z, mem.ψ_szz_z)
end

@kernel function _cpml_scale_x_kernel!(du, @Const(kappa), @Const(ψ))
    i, j = @index(Global, NTuple)
    du[i, j] = du[i, j] / kappa[i] + ψ[i, j]
end

@kernel function _cpml_scale_z_kernel!(du, @Const(kappa), @Const(ψ))
    i, j = @index(Global, NTuple)
    du[i, j] = du[i, j] / kappa[j] + ψ[i, j]
end

function _cpml_scale_x!(du, kappa, ψ)
    device = get_backend(du)
    kernel! = _cpml_scale_x_kernel!(device)
    kernel!(du, kappa, ψ; ndrange=size(du))
end

function _cpml_scale_z!(du, kappa, ψ)
    device = get_backend(du)
    kernel! = _cpml_scale_z_kernel!(device)
    kernel!(du, kappa, ψ; ndrange=size(du))
end

#=============================================================================
# CPML memory variable ODE (for including in extended state)
=============================================================================#

"""
    cpml_memory_eq!(dψ, ψ, derivatives, coeffs)

Update CPML memory variables. This should be called as part of the 
ODE system when using CPML.

The update equation is: dψ/dt = (b-1)/dt * ψ + a * ∂u
For continuous formulation: dψ/dt = -(d/κ + α) * ψ + a_cont * ∂u
"""
@kernel function cpml_memory_update_x_kernel!(dψ, @Const(ψ), @Const(du), @Const(a), @Const(b), dt)
    i, j = @index(Global, NTuple)
    # Continuous form of memory variable derivative
    # dψ/dt ≈ ((b-1)*ψ + a*du) / dt when b = exp(-c*dt)
    dψ[i, j] = (b[i] - 1) / dt * ψ[i, j] + a[i] / dt * du[i, j]
end

@kernel function cpml_memory_update_z_kernel!(dψ, @Const(ψ), @Const(du), @Const(a), @Const(b), dt)
    i, j = @index(Global, NTuple)
    dψ[i, j] = (b[j] - 1) / dt * ψ[i, j] + a[j] / dt * du[i, j]
end

#=============================================================================
# Problem construction
=============================================================================#

"""
    make_problem(parameters::Parameters; s0=nothing, v0=nothing, tspan=(0.0, 2.0))

Create a DynamicalODEProblem for viscoelastic wave propagation.

# Arguments
- `parameters`: Parameters struct with model, grid, and BC configuration
- `s0`: Initial stress field (Nx × Nz × 3), defaults to zeros
- `v0`: Initial velocity field (Nx × Nz × 2), defaults to zeros  
- `tspan`: Time span tuple (t_start, t_end)

# Returns
DynamicalODEProblem compatible with SciML solvers.
"""
function make_problem(parameters::Parameters; s0=nothing, v0=nothing, tspan=(0.0, 2.0))
    Nx = parameters.Nx
    Nz = parameters.Nz
    device = get_backend(parameters.model.b)
    
    if s0 === nothing
        s0 = similar(parameters.model.b, (Nx, Nz, 3))
        s0 .= zero(eltype(s0))
    end
    if v0 === nothing
        v0 = similar(parameters.model.b, (Nx, Nz, 2))
        v0 .= zero(eltype(v0))
    end

    return DynamicalODEProblem(
        stress_eq!, velocity_eq!, 
        s0, v0, 
        tspan, parameters
    )
end

"""
    solve_problem(problem; solver=Tsit5(), kwargs...)

Solve the viscoelastic wave propagation problem.

# Arguments
- `problem`: DynamicalODEProblem from make_problem
- `solver`: ODE solver (default: Tsit5())
- `kwargs...`: Additional arguments passed to solve()

# Returns
ODE solution object.
"""
function solve_problem(problem; solver=Tsit5(), kwargs...)
    return solve(problem, solver; kwargs...)
end