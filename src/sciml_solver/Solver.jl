
"""
    Solver(Nx,Nz,init=zeros)

Preallocated solver struct for elastic wave equations. Solver constructor using grid sizes and an optional initializer. Might be omitted in the future if intermediate steps can be contained within the kernel.

# Arguments
- `Nx,Nz`: Size in x and z direction
- `init`: Initializer for the preallocated matrices.


Examples:`init=zeros` sets all fields to the defualt Julia zeros initializer, i.e. Float64.
- `init=cuzeros` sets all fields to CUDA-allocated zeroed memory, usually Float32 on the device side.



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