using OrdinaryDiffEq
using Zygote
using CUDA

include("Model.jl")
include("Solver.jl")
include("Source.jl")
include("BoundaryCondition.jl")
include(joinpath("..", "Stencil.jl"))

# TODO: Create staggered grid
# TODO: This implementation use intermediate arrays and broadcasting to update fields.
#   Test whether one-step long kernels where global index is used,
#   I -> ∇v[I] -> ds[I], are more efficient than having preallocated arrays and 
#   doing separate update steps for the fields
#   (update ∇v, update ds, in two separate steps).
"""
    _update_ds!(ds, solver::Solver, model::T) where T<:AbstractModel
Update the stress field `ds` from the velocity field using the given `solver` and `model`.

# Arguments
- `ds::Array{T,3}`: The stress field to be updated.
- `solver::Solver`: The solver object containing pre-allocated buffers.
- `model <: AbstractModel`: The model object containing the material properties.

"""
function _update_ds!(ds, solver::Solver, model::T) where T<:AbstractModel
    error("_update_ds! not implemented for $T")
end

"Uses the whole stiffness matrix"
function _update_ds!(ds, solver::Solver, model::TTIModel)

    # Use views for efficient indexing
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11 = model.C11
    C13 = model.C13
    C15 = model.C15
    C33 = model.C33
    C35 = model.C35
    C55 = model.C55

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
function _update_ds!(ds, solver::Solver, model::Union{IsotropicModel,VTIModel})
    # Use views for efficient indexing
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11 = model.C11
    C13 = model.C13
    C33 = model.C33
    C55 = model.C55

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

function stress_eq!(ds, v, s, p, t)

    # Extract views of fields from parameters
    vx = @view v[:, :, 1]
    vz = @view v[:, :, 2]

    model = p.model
    solver = p.solver
    fdm = p.fdm
    bc! = p.bc
    source! = p.source

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


    _update_ds!(ds, solver, model)

    return nothing # in-place update
end


function velocity_eq!(dv, v, s, p, t)
    # Handle the interior leapfrog timestep for velocity
    # Following Fichtner (2012)


    model = p.model
    solver = p.solver
    fdm = p.fdm
    bc! = p.bc
    source! = p.source

    dx_sxx = solver.dx_sxx
    dx_szx = solver.dx_szx # = dz_sxz, but ddx is faster than ddz
    dz_szz = solver.dz_szz

    # buoyancy and forces must be interpolated before they are passed into the function

    # 4. Compute ∇⋅s(t + ½Δt)
    sxx = @view s[:, :, 1]
    szz = @view s[:, :, 2]
    szx = @view s[:, :, 3]

    device = get_backend(dx_sxx)
    ddx!(dx_sxx, sxx, fdm)
    ddx!(dx_szx, szx, fdm) # = dx_sxz
    ddz!(dz_szz, szz, fdm)

    bc! isa PeriodicBC && KA.synchronize(device)

    bc!(dx_sxx)
    bc!(dx_szx)
    bc!(dz_szz)
    KA.synchronize(device)

    # Compute ̇v from momentum balance
    dvx = @view dv[:, :, 1]
    dvz = @view dv[:, :, 2]
    
    dvx = dx_sxx + dx_szx
    dvz = dx_szx + dz_szz
    source!(dvz, t)
    
    b = model.b # buoyancy
    dvx .*= b
    dvz .*= b

    return nothing # in-place update
end

function construct_params(model, solver, fdm, bc, source)
    return (
        model = model, 
        solver = solver, 
        fdm = fdm, 
        bc = bc, 
        source = source)
end


if abspath(PROGRAM_FILE) == @__FILE__
    ## --- testing
    Nx = 512
    Nz = 512
    ρ = 2600.0 # kg/m³
    λ = 36.9e9 # Pa
    μ = 30.65e9 # Pa
    model = IsotropicModel(ρ, λ, μ, Nx, Nz)
    solver = Solver(Nx,Nz)
    # TODO: Look into if FDM and BC can be initialized together (or better initializer for Boundary Conditions)
    fdm = Stencil(8, 1.0)
    bc = ZeroBC(fdm, Nx, Nz)
    source = RickerSource(40, 1, 256, 256)

    p = construct_params(model, solver, fdm, bc, source)

    v0 = zeros(size(solver.dxvx)..., 2)
    s0 = zeros(size(solver.dx_sxx)..., 3)
    tspan = (0,10)

    dv = similar(v0)
    ds = similar(s0)

    println("Stress:")
    @time stress_eq!(ds, v0, s0, p, 0.0)
    @time stress_eq!(ds, v0, s0, p, 0.0)

    println("Velocity:")
    @time velocity_eq!(dv, v0, s0, p, 0.0)
    @time velocity_eq!(dv, v0, s0, p, 0.0)

    # prob = DynamicalODEProblem(stress_eq!, velocity_eq!, s0, v0, tspan, p=p)
    # sol = solve(prob)
end 

