
"""
    Solver(Nx, Nz, init=zeros)

Preallocated solver buffers for elastic wave equations using finite difference stencils.

# Arguments
- `Nx`, `Nz`: Grid dimensions in x and z directions
- `init`: Initializer function for preallocated matrices (default: `zeros`)

# Fields
The solver contains preallocated buffers for:
- `dxvx`, `dzvx`: Spatial derivatives of x-velocity
- `dxvz`, `dzvz`: Spatial derivatives of z-velocity  
- `dx_sxx`, `dx_szx`: Spatial derivatives of stress tensor components
- `dz_szz`: Spatial derivative of zz stress component

# Examples
```Julia
# CPU solver with Float64 arrays
solver = Solver(64, 64, init=zeros)

# GPU solver with CUDA arrays (usually Float32 on device)
using CUDA
solver = Solver(64, 64, init=cuzeros)

# Custom initializer
solver = Solver(128, 128, init=(Nx,Nz) -> randn(Float32, Nx, Nz))
```
"""
struct Solver
    dxvx
    dzvx

    dxvz
    dzvz

    dx_sxx
    dx_szx
    dz_szz
end

Solver(Nx,Nz,init=zeros) = Solver(
    init(Nx,Nz),
    init(Nx,Nz),

    init(Nx,Nz),
    init(Nx,Nz),

    init(Nx,Nz),
    init(Nx,Nz),
    init(Nx,Nz))