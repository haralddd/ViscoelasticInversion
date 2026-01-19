abstract type AbstractBC end

struct ZeroBC <: AbstractBC
    # Target range
    rx0
    rx1
    rz0
    rz1

    function ZeroBC(fdm, Nx, Nz)
        x0 = fdm.x0
        x1 = fdm.x1
        z0 = fdm.z0
        z1 = fdm.z1

        rx0 = 1:x0
        rx1 = (Nx-x1+1):Nx
        rz0 = 1:z0
        rz1 = (Nz-z1+1):Nz

        return new(rx0,rx1,rz0,rz1)
    end
end

struct PeriodicBC <: AbstractBC
    # Target range
    rx0
    rx1
    rz0
    rz1

    # Source range
    sx0
    sx1
    sz0
    sz1

    function PeriodicBC(fdm, Nx, Nz)
        x0 = fdm.x0
        x1 = fdm.x1
        z0 = fdm.z0
        z1 = fdm.z1

        rx0 = 1:fdm.x0
        rx1 = (Nx-x1+1):Nx
        rz0 = 1:fdm.z0
        rz1 = (Nz-z1+1):Nz

        # Calculate indexes for looparound
        sx0 = (Nx-x1-x0+1):(Nx-x1)
        sx1 = (x0+1):(x0+x1)
        sz0 = (Nz-z1-z0+1):(Nz-z1)
        sz1 = (z0+1):(z0+z1)

        return new(rx0,rx1,rz0,rz1,sx0,sx1,sz0,sz1)
    end
end

struct AbsorbingBC <: AbstractBC
    # TODO: Implement
    AbsorbingBC() = error("ABC Not implemented")
end

function (bc::ZeroBC)(du)
    du[bc.rx0, :] .= zero(eltype(du))
    du[bc.rx1, :] .= zero(eltype(du))
    du[:, bc.rz0] .= zero(eltype(du))
    du[:, bc.rz1] .= zero(eltype(du))
end

function (bc::PeriodicBC)(du)
    du[bc.rx0, :] .= du[bc.sx0, :]
    du[bc.rx1, :] .= du[bc.sx1, :]
    du[:, bc.rz0] .= du[:, bc.sz0]
    du[:, bc.rz1] .= du[:, bc.sz1]
    return nothing
end


if abspath(PROGRAM_FILE) == @__FILE__
    Nx = 5
    Nz = 5
    A = rand(1:9, Nx, Nz)
    include("../Stencil.jl")
    fdm = Stencil(2, 0)
    pbc = PeriodicBC(fdm, Nx, Nz)
    zbc = ZeroBC(fdm, Nx, Nz)

    println("Periodic BC test:")
    pbc(A)
    display(A)

    println("Zero BC test:")
    zbc(A)
    display(A)

end