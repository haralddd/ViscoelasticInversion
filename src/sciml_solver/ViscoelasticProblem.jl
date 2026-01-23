


# TODO: Create staggered grid
# TODO: This implementation use intermediate arrays and broadcasting to update fields.
#   Test whether one-step long kernels where global index is used,
#   I -> ∇v[I] -> ds[I], are more efficient than having preallocated arrays and 
#   doing separate update steps for the fields
#   (update ∇v, update ds, in two separate steps).
"""
    _update_ds!(ds, prealloc::prealloc, model::T) where T<:AbstractModel
Update the stress field `ds` from the velocity field using the given `prealloc` and `model`.

# Arguments
- `ds::Array{T,3}`: The stress field to be updated.
- `prealloc::prealloc`: The prealloc object containing pre-allocated buffers.
- `model <: AbstractModel`: The model object containing the material properties.

"""
function _update_ds!(ds, prealloc::Preallocated, model::T) where T<:AbstractModel
    error("_update_ds! not implemented for $T")
end

"Uses the whole stiffness matrix"
function _update_ds!(ds, prealloc::Preallocated, model::TTIModel)

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

    dxvx = prealloc.dxvx
    dzvx = prealloc.dzvx
    dxvz = prealloc.dxvz
    dzvz = prealloc.dzvz

    # 2. Compute ̇s(t) from ̇ε(t)
    @. dsxx = C11 * dxvx + C13 * dzvz + C15 * (dzvx + dxvz)
    @. dszz = C13 * dxvx + C33 * dzvz + C35 * (dzvx + dxvz)
    @. dsxz = C15 * dxvx + C35 * dzvz + C55 * (dzvx + dxvz)
end

"Uses the VTI parts of the stiffness matrix"
function _update_ds!(ds, prealloc::Preallocated, model::Union{IsotropicModel,VTIModel})
    # Use views for efficient indexing
    dsxx = @view ds[:, :, 1]
    dszz = @view ds[:, :, 2]
    dsxz = @view ds[:, :, 3]

    C11 = model.C11
    C13 = model.C13
    C33 = model.C33
    C55 = model.C55

    dxvx = prealloc.dxvx
    dzvx = prealloc.dzvx
    dxvz = prealloc.dxvz
    dzvz = prealloc.dzvz

    # 2. Compute ̇s(t) from ̇ε(t)
    @. dsxx = C11 * dxvx + C13 * dzvz
    @. dszz = C13 * dxvx + C33 * dzvz
    @. dsxz = C55 * (dzvx + dxvz)
end

# TODO: Impl absorbing boundary conditions with forward and backward differences

function stress_eq!(ds, s, v, p::Parameters, t)

    # Extract views of fields from parameters
    vx = @view v[:, :, 1]
    vz = @view v[:, :, 2]

    model = p.model
    prealloc = p.prealloc
    fdm = p.fdm
    bc! = p.bc
    source! = p.source

    dxvx = prealloc.dxvx
    dzvx = prealloc.dzvx
    dxvz = prealloc.dxvz
    dzvz = prealloc.dzvz

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


    _update_ds!(ds, prealloc, model)

    return nothing # in-place update
end


function velocity_eq!(dv, s, v, p::Parameters, t)
    # Handle the interior leapfrog timestep for velocity
    # Following Fichtner (2012)


    model = p.model
    prealloc = p.prealloc
    fdm = p.fdm
    bc! = p.bc
    source! = p.source

    dx_sxx = prealloc.dx_sxx
    dx_szx = prealloc.dx_szx # = dz_sxz, but ddx is faster than ddz
    dz_szz = prealloc.dz_szz

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
    
    @. dvx = dx_sxx + dx_szx
    @. dvz = dx_szx + dz_szz
    source!(dvz, t)
    KA.synchronize(device)
    
    b = model.b # buoyancy
    dvx .*= b
    dvz .*= b

    return nothing # in-place update
end


function make_problem(parameters::Parameters, s0=nothing, v0=nothing, tspan=(0.0, 2.0))
    Nx = parameters.Nx
    Nz = parameters.Nz
    if s0 === nothing
        s0 = similar(parameters.model.b, (Nx, Nz, 3))
        s0 .= zero(eltype(s0))
    end
    if v0 === nothing
        v0 = similar(parameters.model.b, (Nx, Nz, 2))
        v0 .= zero(eltype(v0))
    end

    return DynamicalODEProblem(
        stress_eq!, velocity_eq!, 
        s0, v0, 
        tspan, parameters
    )
end


function solve_problem(problem)
    saved_values = SavedValues(Float64, Array{Float32,3})
    cb = SavingCallback((u,t,i)-> Array(u.x), saved_values, saveat=0.0:0.01:2.0)
    solve(problem, callback=cb, tstops=[1.0], save_on=false, save_start=false, save_end=false)
    return saved_values
end