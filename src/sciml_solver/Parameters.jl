# Wrapper struct for parameters to avoid large tuples in ViscoelasticProblem
"""
    Parameters(kwargs...)

Parameter setup struct for the ViscoelasticProblem solver

# Fields
- `model`: Model object
- `prealloc`: Preallocated storage
- `fdm`: Finite difference stencil
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
    fdm
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
    
    prealloc = Preallocated(Nx, Nz, device=device)
    fdm = Stencil(fd_order_x, fd_order_z, dx, dz, device=device)
    bc = get_bc(bc_type, fdm, Nx, Nz)
    return Parameters(Nx, Nz, model, prealloc, fdm, bc, source)
end