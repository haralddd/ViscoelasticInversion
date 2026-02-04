"""
    CPML - Convolutional Perfectly Matched Layer implementation

This module implements the Complex Frequency-Shifted Convolutional PML (CFS-CPML)
for absorbing boundary conditions in viscoelastic wave propagation.

The CPML modifies spatial derivatives in the PML region using memory variables
that are updated alongside the main wave equations.

Reference: Komatitsch & Martin (2007), "An unsplit convolutional perfectly 
matched layer improved at grazing incidence for the seismic wave equation"
"""

"""
    CPMLConfig

Configuration for CPML regions. Defines which boundaries have PML and their parameters.

# Fields
- `thickness::Int`: Number of grid points in PML region
- `d0::Float64`: Maximum damping coefficient  
- `alpha_max::Float64`: CFS alpha parameter (for evanescent wave absorption)
- `kappa_max::Float64`: CFS kappa parameter (coordinate stretching)
- `power::Float64`: Polynomial order for damping profile
- `left::Bool`, `right::Bool`, `bottom::Bool`: Which boundaries have PML
- `top::Bool`: Top boundary (typically false for free surface)
"""
struct CPMLConfig
    thickness::Int
    d0::Float64
    alpha_max::Float64
    kappa_max::Float64
    power::Float64
    left::Bool
    right::Bool
    top::Bool
    bottom::Bool
end

function CPMLConfig(;
    thickness=10,
    d0=nothing,
    alpha_max=0.0,
    kappa_max=1.0,
    power=2.0,
    left=true,
    right=true,
    top=false,  # Free surface by default
    bottom=true,
    # For automatic d0 calculation
    vmax=nothing,
    dx=nothing,
    reflection_coef=1e-5
)
    if d0 === nothing
        if vmax === nothing || dx === nothing
            error("Must provide either d0 or (vmax and dx) for automatic damping calculation")
        end
        # Optimal d0 from Collino & Tsogka (2001)
        d0 = -(power + 1) * vmax * log(reflection_coef) / (2 * thickness * dx)
    end
    return CPMLConfig(thickness, d0, alpha_max, kappa_max, power, left, right, top, bottom)
end

"""
    CPMLCoefficients

Precomputed CPML damping coefficients for a specific grid.
These are the a, b coefficients used in the recursive convolution.
"""
struct CPMLCoefficients{T}
    # Damping profiles (full grid, zero in interior)
    d_x::T      # d(x) profile in x-direction
    d_z::T      # d(z) profile in z-direction
    alpha_x::T  # alpha(x) CFS parameter
    alpha_z::T  # alpha(z) CFS parameter
    kappa_x::T  # kappa(x) stretching parameter
    kappa_z::T  # kappa(z) stretching parameter
    
    # Recursive update coefficients: ψ_new = b*ψ + a*∂u
    a_x::T      # a coefficient for x-direction
    a_z::T      # a coefficient for z-direction
    b_x::T      # b coefficient for x-direction
    b_z::T      # b coefficient for z-direction
end

"""
Compute damping profile value at normalized position p ∈ [0, 1]
where p=0 is the interface and p=1 is the outer boundary.
"""
function _damping_profile(p::Real, d0::Real, power::Real)
    return d0 * p^power
end

function _alpha_profile(p::Real, alpha_max::Real)
    # Alpha increases towards interface (opposite to d)
    return alpha_max * (1 - p)
end

function _kappa_profile(p::Real, kappa_max::Real)
    return 1 + (kappa_max - 1) * p
end

"""
    CPMLCoefficients(config::CPMLConfig, Nx, Nz, dx, dz, dt; device=CPU())

Construct CPML coefficients for the given grid configuration.
"""
function CPMLCoefficients(config::CPMLConfig, Nx, Nz, dx, dz, dt; device=CPU())
    F = preferred_float(device)
    L = config.thickness
    
    # Initialize arrays on CPU first, then copy to device
    d_x = zeros(F, Nx)
    d_z = zeros(F, Nz)
    alpha_x = zeros(F, Nx)
    alpha_z = zeros(F, Nz)
    kappa_x = ones(F, Nx)
    kappa_z = ones(F, Nz)
    
    # Left boundary (x = 1 to L)
    if config.left
        for i in 1:L
            p = (L - i + 0.5) / L  # Normalized distance from interface
            d_x[i] = _damping_profile(p, config.d0, config.power)
            alpha_x[i] = _alpha_profile(p, config.alpha_max)
            kappa_x[i] = _kappa_profile(p, config.kappa_max)
        end
    end
    
    # Right boundary (x = Nx-L+1 to Nx)
    if config.right
        for i in 1:L
            p = (i - 0.5) / L  # Normalized distance from interface
            idx = Nx - L + i
            d_x[idx] = _damping_profile(p, config.d0, config.power)
            alpha_x[idx] = _alpha_profile(p, config.alpha_max)
            kappa_x[idx] = _kappa_profile(p, config.kappa_max)
        end
    end
    
    # Top boundary (z = 1 to L) - typically disabled for free surface
    if config.top
        for j in 1:L
            p = (L - j + 0.5) / L
            d_z[j] = _damping_profile(p, config.d0, config.power)
            alpha_z[j] = _alpha_profile(p, config.alpha_max)
            kappa_z[j] = _kappa_profile(p, config.kappa_max)
        end
    end
    
    # Bottom boundary (z = Nz-L+1 to Nz)
    if config.bottom
        for j in 1:L
            p = (j - 0.5) / L
            idx = Nz - L + j
            d_z[idx] = _damping_profile(p, config.d0, config.power)
            alpha_z[idx] = _alpha_profile(p, config.alpha_max)
            kappa_z[idx] = _kappa_profile(p, config.kappa_max)
        end
    end
    
    # Compute recursive convolution coefficients
    # For the update: ψ_new = b*ψ + a*∂u
    # where b = exp(-(d/κ + α)Δt) and a = d/(κ(d + κα))(b - 1)
    a_x = zeros(F, Nx)
    a_z = zeros(F, Nz)
    b_x = zeros(F, Nx)
    b_z = zeros(F, Nz)
    
    for i in 1:Nx
        if d_x[i] > 0 || alpha_x[i] > 0
            dk = d_x[i] / kappa_x[i]
            b_x[i] = exp(-(dk + alpha_x[i]) * dt)
            if abs(d_x[i]) > eps(F)
                a_x[i] = d_x[i] / (kappa_x[i] * (d_x[i] + kappa_x[i] * alpha_x[i])) * (b_x[i] - 1)
            end
        end
    end
    
    for j in 1:Nz
        if d_z[j] > 0 || alpha_z[j] > 0
            dk = d_z[j] / kappa_z[j]
            b_z[j] = exp(-(dk + alpha_z[j]) * dt)
            if abs(d_z[j]) > eps(F)
                a_z[j] = d_z[j] / (kappa_z[j] * (d_z[j] + kappa_z[j] * alpha_z[j])) * (b_z[j] - 1)
            end
        end
    end
    
    # Copy to device
    d_x_dev = allocate(device, F, Nx)
    d_z_dev = allocate(device, F, Nz)
    alpha_x_dev = allocate(device, F, Nx)
    alpha_z_dev = allocate(device, F, Nz)
    kappa_x_dev = allocate(device, F, Nx)
    kappa_z_dev = allocate(device, F, Nz)
    a_x_dev = allocate(device, F, Nx)
    a_z_dev = allocate(device, F, Nz)
    b_x_dev = allocate(device, F, Nx)
    b_z_dev = allocate(device, F, Nz)
    
    copyto!(d_x_dev, d_x)
    copyto!(d_z_dev, d_z)
    copyto!(alpha_x_dev, alpha_x)
    copyto!(alpha_z_dev, alpha_z)
    copyto!(kappa_x_dev, kappa_x)
    copyto!(kappa_z_dev, kappa_z)
    copyto!(a_x_dev, a_x)
    copyto!(a_z_dev, a_z)
    copyto!(b_x_dev, b_x)
    copyto!(b_z_dev, b_z)
    
    return CPMLCoefficients(
        d_x_dev, d_z_dev,
        alpha_x_dev, alpha_z_dev,
        kappa_x_dev, kappa_z_dev,
        a_x_dev, a_z_dev,
        b_x_dev, b_z_dev
    )
end

"""
    CPMLMemory

Memory variables for CPML. These are updated as part of the ODE system.

For velocity-stress formulation, we need memory variables for:
- ∂vx/∂x, ∂vx/∂z, ∂vz/∂x, ∂vz/∂z (for stress update)
- ∂σxx/∂x, ∂σxz/∂x, ∂σxz/∂z, ∂σzz/∂z (for velocity update)
"""
struct CPMLMemory{T}
    # Memory for velocity derivatives (used in stress equation)
    ψ_vx_x::T   # Memory for ∂vx/∂x
    ψ_vx_z::T   # Memory for ∂vx/∂z
    ψ_vz_x::T   # Memory for ∂vz/∂x
    ψ_vz_z::T   # Memory for ∂vz/∂z
    
    # Memory for stress derivatives (used in velocity equation)
    ψ_sxx_x::T  # Memory for ∂σxx/∂x
    ψ_sxz_x::T  # Memory for ∂σxz/∂x (= ∂σzx/∂x)
    ψ_sxz_z::T  # Memory for ∂σxz/∂z
    ψ_szz_z::T  # Memory for ∂σzz/∂z
end

function CPMLMemory(Nx, Nz; device=CPU())
    F = preferred_float(device)
    
    ψ_vx_x = KA.zeros(device, F, Nx, Nz)
    ψ_vx_z = KA.zeros(device, F, Nx, Nz)
    ψ_vz_x = KA.zeros(device, F, Nx, Nz)
    ψ_vz_z = KA.zeros(device, F, Nx, Nz)
    
    ψ_sxx_x = KA.zeros(device, F, Nx, Nz)
    ψ_sxz_x = KA.zeros(device, F, Nx, Nz)
    ψ_sxz_z = KA.zeros(device, F, Nx, Nz)
    ψ_szz_z = KA.zeros(device, F, Nx, Nz)
    
    return CPMLMemory(
        ψ_vx_x, ψ_vx_z, ψ_vz_x, ψ_vz_z,
        ψ_sxx_x, ψ_sxz_x, ψ_sxz_z, ψ_szz_z
    )
end

"""
    CPMLBC <: AbstractBC

CPML boundary condition that modifies derivatives using memory variables.
"""
struct CPMLBC <: AbstractBC
    config::CPMLConfig
    coeffs::CPMLCoefficients
    memory::CPMLMemory
end

function CPMLBC(config::CPMLConfig, Nx, Nz, dx, dz, dt; device=CPU())
    coeffs = CPMLCoefficients(config, Nx, Nz, dx, dz, dt; device=device)
    memory = CPMLMemory(Nx, Nz; device=device)
    return CPMLBC(config, coeffs, memory)
end

"""
Kernel to apply CPML correction to a derivative and update memory variable.

The CPML-modified derivative is:
    ∂u/∂x_cpml = (1/κ) * ∂u/∂x + ψ

And the memory variable update is:
    ψ_new = b*ψ + a*∂u/∂x
"""
@kernel function cpml_update_x_kernel!(du_dx, @Const(ψ), dψ, @Const(a), @Const(b), @Const(kappa))
    i, j = @index(Global, NTuple)
    
    # Get coefficients for this x-position
    a_i = a[i]
    b_i = b[i]
    κ_i = kappa[i]
    
    # Current derivative and memory value
    du_val = du_dx[i, j]
    ψ_val = ψ[i, j]
    
    # CPML-modified derivative: (1/κ)*∂u + ψ
    du_dx[i, j] = du_val / κ_i + ψ_val
    
    # Memory variable rate: dψ/dt = b*ψ/dt + a*∂u (stored as increment for explicit update)
    # For the ODE formulation, we store the time derivative of ψ
    dψ[i, j] = (b_i - 1) / 1.0 * ψ_val + a_i * du_val  # Approximate for small dt
end

@kernel function cpml_update_z_kernel!(du_dz, @Const(ψ), dψ, @Const(a), @Const(b), @Const(kappa))
    i, j = @index(Global, NTuple)
    
    # Get coefficients for this z-position
    a_j = a[j]
    b_j = b[j]
    κ_j = kappa[j]
    
    # Current derivative and memory value
    du_val = du_dz[i, j]
    ψ_val = ψ[i, j]
    
    # CPML-modified derivative
    du_dz[i, j] = du_val / κ_j + ψ_val
    
    # Memory variable rate
    dψ[i, j] = (b_j - 1) / 1.0 * ψ_val + a_j * du_val
end

"""
    apply_cpml_x!(du_dx, ψ, dψ, coeffs)

Apply CPML correction to x-derivative and compute memory variable update rate.
"""
function apply_cpml_x!(du_dx, ψ, dψ, coeffs::CPMLCoefficients)
    device = get_backend(du_dx)
    Nx, Nz = size(du_dx)
    
    kernel! = cpml_update_x_kernel!(device)
    kernel!(du_dx, ψ, dψ, coeffs.a_x, coeffs.b_x, coeffs.kappa_x; ndrange=(Nx, Nz))
    
    return nothing
end

"""
    apply_cpml_z!(du_dz, ψ, dψ, coeffs)

Apply CPML correction to z-derivative and compute memory variable update rate.
"""
function apply_cpml_z!(du_dz, ψ, dψ, coeffs::CPMLCoefficients)
    device = get_backend(du_dz)
    Nx, Nz = size(du_dz)
    
    kernel! = cpml_update_z_kernel!(device)
    kernel!(du_dz, ψ, dψ, coeffs.a_z, coeffs.b_z, coeffs.kappa_z; ndrange=(Nx, Nz))
    
    return nothing
end

"""
    FreeSurfaceBC

Free surface boundary condition for the top boundary.
Implements stress imaging method (Levander, 1988).
"""
struct FreeSurfaceBC
    depth::Int  # Number of ghost points to mirror
end

FreeSurfaceBC() = FreeSurfaceBC(1)

@kernel function apply_free_surface_stress_kernel!(σxz, σzz, depth)
    i = @index(Global)
    
    # Mirror stress antisymmetrically for free surface at z=1
    for d in 1:depth
        σxz[i, d] = -σxz[i, 2*depth - d + 2]
        σzz[i, d] = -σzz[i, 2*depth - d + 2]
    end
end

function apply_free_surface!(s, bc::FreeSurfaceBC)
    device = get_backend(s)
    Nx = size(s, 1)
    
    σxz = @view s[:, :, 3]
    σzz = @view s[:, :, 2]
    
    kernel! = apply_free_surface_stress_kernel!(device)
    kernel!(σxz, σzz, bc.depth; ndrange=Nx)
    
    return nothing
end
