using OrdinaryDiffEq
using Zygote
using CUDA

include("ElasticModel.jl")
include("Solver.jl")
include(joinpath("..", "Stencil.jl"))

# TODO: This implementation use intermediate arrays and broadcasting to update fields.
#   Test whether one-step long kernels where global index 
#   I -> ∇v[I] -> ds[I] are more efficient than having preallocated arrays and 
#   doing separate update steps for the fields, 
#   i.e. update ∇v, update ds, in two steps.
"""
Update the strain field `ds` from the velocity field using the given `solver` and `medium`.

# Arguments
- `ds::Array{T,3}`: The strain field to be updated.
- `solver::Solver`: The solver object containing pre-allocated buffers.
- `medium <: AbstractMedium`: The medium object containing the material properties.

"""
function _update_ds!(ds, solver::Solver, medium::T) where T<:AbstractMedium
    error("_update_ds! not implemented for $T")
end

"Uses the whole stiffness matrix"
function _update_ds!(ds, solver::Solver, medium::TTIMedium)

    # Use views for efficient indexing
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11 = medium.C11
    C13 = medium.C13
    C15 = medium.C15
    C33 = medium.C33
    C35 = medium.C35
    C55 = medium.C55

    dxvx = solver.dxvx
    dzvx = solver.dzvx
    dxvz = solver.dxvz
    dzvz = solver.dzvz

    # 2. Compute ̇s(t) from ̇ε(t)
    @. dsxx = C11 * dxvx + C13 * dzvz + C15 * (dzvx + dxvz)
    @. dszz = C13 * dxvx + C33 * dzvz + C35 * (dzvx + dxvz)
    @. dsxz = C15 * dxvx + C35 * dzvz + C55 * (dzvx + dxvz)
end

"Uses the VTI parts of the stiffness matrix"
function _update_ds!(ds, solver::Solver, medium::Union{IsotropicMedium,VTIMedium})
    # Use views for efficient indexing
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11 = medium.C11
    C13 = medium.C13
    C33 = medium.C33
    C55 = medium.C55

    dxvx = solver.dxvx
    dzvx = solver.dzvx
    dxvz = solver.dxvz
    dzvz = solver.dzvz

    # 2. Compute ̇s(t) from ̇ε(t)
    @. dsxx = C11 * dxvx + C13 * dzvz
    @. dszz = C13 * dxvx + C33 * dzvz
    @. dsxz = C55 * (dzvx + dxvz)
end



# TODO: Impl absorbing boundary conditions with forward and backward differences

function strain_eq!(ds, v, s, p, t)

    # Extract views of fields from parameters
    vx = @view v[:, :, 1]
    vz = @view v[:, :, 2]

    medium = p.medium
    solver = p.solver
    fdm = p.fdm
    bc! = p.bc!
    source! = p.source!

    dxvx = solver.dxvx
    dzvx = solver.dzvx
    dxvz = solver.dxvz
    dzvz = solver.dzvz

    device = get_backend(dxvx)
    ddx!(dxvx, vx, fdm)
    ddz!(dzvx, vx, fdm)
    ddx!(dxvz, vz, fdm)
    ddz!(dzvz, vz, fdm)

    # Only needed for periodic boundary conditions
    bc! isa PeriodicBC && KA.synchronize(device)

    bc!(dxvx)
    bc!(dzvx)
    bc!(dxvz)
    bc!(dzvz)
    KA.synchronize(device)


    _update_ds!(ds, solver, medium)
    source!(ds, t)

    return nothing # in-place update
end


function velocity_eq!(dv, v, s, p, t)
    # Handle the interior leapfrog timestep for velocity
    # Following Fichtner (2012)

    dvx = @view dv[:, :, 1]
    dvz = @view dv[:, :, 2]


    medium = p.medium
    solver = p.solver
    fdm = p.fdm
    bc! = p.bc_func!
    source! = p.source_func!

    dx_sxx = solver.dx_sxx
    dx_szx = solver.dx_szx # = dz_sxz, but ddx is faster than ddz
    dz_szz = solver.dz_szz

    # buoyancy and forces must be interpolated before they are passed into the function
    bx = medium
    αz = αzs[m, n]
    fx = fxs[m, n]
    fz = fzs[m, n]

    sxx = @view s[:, :, 1]
    szz = @view s[:, :, 2]
    szx = @view s[:, :, 3]

    # 4. Compute ∇⋅s(t + ½Δt)
    ddx!(dx_sxx, sxx, fdm)
    ddx!(dx_szx, szx, fdm) # = dx_sxz
    ddz!(dz_szz, szz, fdm)

    # Compute ̇v from momentum balance
    ∂vx_∂t = αx * (∂sxx_∂x + ∂szx_∂z + fx)
    ∂vz_∂t = αz * (∂sxz_∂x + ∂szz_∂z + fz)


    return nothing # in-place update
end

if abspath(PROGRAM_FILE) == @__FILE__
    ## --- testing
    Nx = 1000
    Nz = 1000
    C11 = C55 = fill(98.2, Nx, Nz) # GPa
    C13 = fill(39.9, Nx, Nz)

end

