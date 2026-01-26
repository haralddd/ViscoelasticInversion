# Wrapper struct for parameters to avoid large tuples in ViscoelasticProblem
"""
    Parameters(kwargs...)

Parameter setup struct for the ViscoelasticProblem solver

# Fields
- `model`: Model object
- `prealloc`: Preallocated storage
- `fdm_stress`: Staggered stencil for stress updates (accesses velocity at half-grid)
- `fdm_velocity`: Staggered stencil for velocity updates (accesses stress at integer grid)
- `bc`: Boundary conditions
- `source`: Source object

# Parameters
- `Nx::Int`: Number of grid points in x-direction
- `Nz::Int`: Number of grid points in z-direction
- `dx::Float64`: Grid spacing in x-direction
- `dz::Float64`: Grid spacing in z-direction
- `fd_order_x::Int`: Finite difference order
- `fd_order_z::Int`: Finite difference order
- `bc <:AbstractBC`: Boundary conditions
- `source <:AbstractSource`: Source object
- `device <:KernelAbstractions.Backend`: Device to use for computation

# Example
```julia
params = Parameters(Nx=100, Nz=50, dx=10.0, dz=10.0)
```
"""
struct Parameters
    Nx
    Nz
    model
    prealloc
    fdm_plus
    fdm_minus
    bc
    source
end

function Parameters(;kwargs...)
    # Set up staggered grid and do initialization based on parameters

    Nx = get(kwargs, :Nx, 64)
    Nz = get(kwargs, :Nz, 64)
    dx = get(kwargs, :dx, 10.0)
    dz = get(kwargs, :dz, 10.0)

    fd_order_x = get(kwargs, :fd_order_x, 8)
    fd_order_z = get(kwargs, :fd_order_z, 8)
    bc_type = get(kwargs, :bc, :periodic)
    source = get(kwargs, :source, RickerSource(40.0,0.2,Nx÷2,Nz÷2))
    device = get(kwargs, :device, CPU())
    
    # Create default isotropic model if not provided
    model = get(kwargs, :model, IsotropicModel(2000.0, 1e9, 1e9, Nx, Nz, device=device))
    
    prealloc = Preallocated(Nx, Nz, device=device)
    fdm_plus = Stencil(fd_order_x, fd_order_z, dx, dz, Val(:staggered_plus), device=device)
    fdm_minus = Stencil(fd_order_x, fd_order_z, dx, dz, Val(:staggered_minus), device=device)
    bc = ViscoelasticInversion.get_bc(bc_type, fdm_plus, Nx, Nz)
    return Parameters(Nx, Nz, model, prealloc, fdm_plus, fdm_minus, bc, source)
end