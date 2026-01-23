# Wrapper struct for parameters to avoid large tuples in ViscoelasticProblem
"""
    Parameters(kwargs...)

Parameter setup struct for the ViscoelasticProblem solver

# Fields
- `model`: Model object
- `prealloc`: Preallocated storage
- `fdm_x`: Finite difference method in x-direction
- `fdm_z`: Finite difference method in z-direction
- `bc`: Boundary conditions
- `source`: Source object

# Parameters
- `Nx::Int`: Number of grid points in x-direction
- `Nz::Int`: Number of grid points in z-direction
- `dx::Float64`: Grid spacing in x-direction
- `dz::Float64`: Grid spacing in z-direction
- `fd_order_x::Int`: Finite difference order
- `fd_order_z::Int`: Finite difference order
- `bc::T where T<:AbstractBC`: Boundary conditions
- `source::T where T<:AbstractSource`: Source object

# Example
```julia
params = Parameters(Nx=100, Nz=50, dx=10.0, dz=10.0)
```
"""
struct Parameters
    model
    prealloc
    fdm_x
    fdm_z
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
    bc = get(kwargs, :bc, PeriodicBC())
    source = get(kwargs, :source, Source())
    
    prealloc = Preallocated(Nx, Nz)
    fdm_x = Stencil(fd_order_x, dx)
    fdm_z = Stencil(fd_order_z, dz)
    return Parameters(model, prealloc, fdm_x, fdm_z, bc, source)
end