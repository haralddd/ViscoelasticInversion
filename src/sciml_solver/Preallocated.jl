"""
    Preallocated(Nx, Nz; device=CPU())

Preallocated buffers for elastic wave equations using finite difference stencils.

# Arguments
- `Nx`, `Nz`: Grid dimensions in x and z directions
- `device`: Device for allocation (CPU or GPU)

# Fields
The struct contains preallocated buffers for:
- `dxvx`, `dzvx`: Spatial derivatives of x-velocity
- `dxvz`, `dzvz`: Spatial derivatives of z-velocity  
- `dx_sxx`, `dx_szx`,`dz_szz`: Spatial derivatives of stress tensor components

# Examples
```Julia
# CPU Preallocated with Float64 arrays
preallocated = Preallocated(64, 64, device=CPU())

# GPU Preallocated with CUDA arrays (usually Float32 on device)
using CUDA
preallocated = Preallocated(64, 64, device=CUDA())

# Make preallocated struct based on initial stress or velocity
preallocated = Preallocated(s0)
```
"""
struct Preallocated
    # TODO: Reuse the storage locations, since equations are split step
    dxvx
    dzvx

    dxvz
    dzvz

    dx_sxx
    dx_szx
    dz_szz
end

function Preallocated(Nx, Nz; device=CPU())
    T = preferred_float(device)
    dxvx = KA.zeros(device, T, Nx, Nz)
    dzvx = KA.zeros(device, T, Nx, Nz)
    dxvz = KA.zeros(device, T, Nx, Nz)
    dzvz = KA.zeros(device, T, Nx, Nz)
    dx_sxx = KA.zeros(device, T, Nx, Nz)
    dx_szx = KA.zeros(device, T, Nx, Nz)
    dz_szz = KA.zeros(device, T, Nx, Nz)
    
    return Preallocated(dxvx, dzvx, dxvz, dzvz, dx_sxx, dx_szx, dz_szz)
end

function Preallocated(s0)
    Nx, Nz = size(s0)[1], size(s0)[2]
    device = get_backend(s0)
    
    return Preallocated(Nx, Nz; device=device)
end