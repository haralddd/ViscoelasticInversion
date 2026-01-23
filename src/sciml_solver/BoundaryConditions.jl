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

function (bc::PeriodicBC)(du)
    # Apply periodic boundary conditions to derivative array
    # This handles the boundary regions that aren't computed by the main stencil
    
    Nx, Nz = size(du)
    fdm = bc.fdm
    device = get_backend(du)
    
    # Left boundary (copy from right)
    kernel_left = _ddx_kernel_periodic_left!(device)
    kernel_left(du, du, fdm.xgrid, fdm.xcoefs, Nx, Nz; ndrange=(fdm.x0, Nz))
    
    # Right boundary (copy from left)  
    kernel_right = _ddx_kernel_periodic_right!(device)
    kernel_right(du, du, fdm.xgrid, fdm.xcoefs, Nx, Nz; ndrange=(fdm.x1, Nz))
    
    # Top boundary (copy from bottom)
    kernel_top = _ddx_kernel_periodic_top!(device)
    kernel_top(du, du, fdm.zgrid, fdm.zcoefs, Nx, Nz; ndrange=(Nx, fdm.z0))
    
    # Bottom boundary (copy from top)
    kernel_bottom = _ddx_kernel_periodic_bottom!(device)
    kernel_bottom(du, du, fdm.zgrid, fdm.zcoefs, Nx, Nz; ndrange=(Nx, fdm.z1))
    
    return nothing
end


struct AbsorbingBC <: AbstractBC
    # TODO: Implement
    AbsorbingBC() = error("ABC Not implemented")
end


function get_bc(bc_type::Symbol, fdm, Nx, Nz)
    if bc_type == :dirichlet
        return DirichletBC(fdm, Nx, Nz)
    elseif bc_type == :neumann
        return NeumannBC(fdm, Nx, Nz)
    elseif bc_type == :periodic
        return PeriodicBC(fdm, Nx, Nz)
    elseif bc_type == :absorbing
        return AbsorbingBC()
    else
        error("Unknown boundary condition type: $bc_type")
    end
end
