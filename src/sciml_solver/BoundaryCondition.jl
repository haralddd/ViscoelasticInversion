abstract type AbstractBC end

struct DirichletBC <: AbstractBC
    DirichletBC() = new()
    DirichletBC(fdm, Nx, Nz) = DirichletBC() # To match other constructors
end

function (bc::DirichletBC)(du)
    # no-op as the boundaries are fixed in value
    return nothing
end

struct NeumannBC <: AbstractBC
    val
    # Target range
    rx0
    rx1
    rz0
    rz1

    function NeumannBC(fdm, Nx, Nz; val=0.0)
        x0 = fdm.x0
        x1 = fdm.x1
        z0 = fdm.z0
        z1 = fdm.z1

        rx0 = 1:x0
        rx1 = (Nx-x1+1):Nx
        rz0 = 1:z0
        rz1 = (Nz-z1+1):Nz

        return new(val, rx0,rx1,rz0,rz1)
    end
end

function (bc::NeumannBC)(du)
    du[bc.rx0, :] .= bc.val
    du[bc.rx1, :] .= bc.val
    du[:, bc.rz0] .= bc.val
    du[:, bc.rz1] .= bc.val
    return nothing
end

struct PeriodicBC <: AbstractBC
    fdm
    Nx
    Nz
    
    function PeriodicBC(fdm, Nx, Nz)
        new(fdm, Nx, Nz)
    end
end


struct AbsorbingBC <: AbstractBC
    # TODO: Implement
    AbsorbingBC() = error("ABC Not implemented")
end


if abspath(PROGRAM_FILE) == @__FILE__
    Nx = 10
    Nz = 10
    A = rand(1:9, Nx, Nz)
    include("../Stencil.jl")
    fdm = Stencil(4, 0)
    pbc = PeriodicBC(fdm, Nx, Nz)
    zbc = ZeroBC(fdm, Nx, Nz)

    println("Periodic BC test:")
    pbc(A)
    display(A)

    println("Zero BC test:")
    zbc(A)
    display(A)


    println("Testing CUDA array")
    using CUDA
    Acu = cu(A)
    fdm_gpu = Stencil(4, 0, device=get_backend(Acu))
    println("Periodic BC test:")
    pbc(Acu)

    println("Zero BC test:")
    zbc(Acu)



end