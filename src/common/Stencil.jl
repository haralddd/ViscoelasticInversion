function _stencil(x::AbstractVector{<:Real}, x₀::Real, m::Integer)
    ℓ = 0:length(x)-1
    m in ℓ || throw(ArgumentError("order $m ∉ $ℓ"))
    A = @. (x' - x₀)^ℓ / factorial(ℓ)
    return A \ (ℓ .== m) # vector of weights w
end

"""
    Stencil(order, h; device=CPU())
    Stencil(xorder, zorder, Δx, Δz; device=CPU())
    Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())

High-order finite difference stencil for computing spatial derivatives.

# Constructors
- `Stencil(order, h; device=CPU())`: Creates isotropic stencil with given order and spacing
- `Stencil(xorder, zorder, Δx, Δz; device=CPU())`: Creates anisotropic stencil with different orders and spacings
- `Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())`: Creates stencil from custom grid points

# Arguments
- `order`: Finite difference order (must be even)
- `xorder`, `zorder`: Orders in x and z directions (must be even)
- `h`, `Δx`, `Δz`: Grid spacing
- `xgrid`, `zgrid`: Grid point offsets
- `x0`, `z0`: Reference point indices
- `device`: Choose device to allocate indices and coefficients to

# Examples
```Julia
# 8th-order isotropic stencil with unit spacing
stencil = Stencil(8, 1.0)

# 8th-order in x, 4th-order in z with different spacings
stencil = Stencil(8, 4, 1.0, 0.5)

# Custom stencil from grid points
xgrid = [-2, -1, 1, 2]
zgrid = [-2, -1, 1, 2]
stencil = Stencil(xgrid, zgrid, 0, 0, 1.0, 1.0)
```
"""
struct Stencil
    xgrid
    zgrid

    xcoefs
    zcoefs

    # Define interior bounds
    x0
    z0
    x1
    z1

    Stencil(xgrid, zgrid, xcoefs, zcoefs, x0, z0, x1, z1) = new(xgrid, zgrid, xcoefs, zcoefs, x0, z0, x1, z1)

    function Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())

        I = preferred_int(device)
        F = preferred_float(device)
        _xgrid = allocate(device, I, size(xgrid))
        _zgrid = allocate(device, I, size(zgrid))
        _xcoefs = similar(_xgrid, F)
        _zcoefs = similar(_zgrid, F)

        xcoefs = _stencil(Rational.(xgrid), x0, 1) ./ Δx
        zcoefs = _stencil(Rational.(zgrid), z0, 1) ./ Δz

        copyto!(_xgrid, Vector{I}(xgrid))
        copyto!(_zgrid, Vector{I}(zgrid))
        copyto!(_xcoefs, Vector{F}(xcoefs))
        copyto!(_zcoefs, Vector{F}(zcoefs))

        x0 = I(abs(min(minimum(xgrid), 0)))
        z0 = I(abs(min(minimum(zgrid), 0)))
        x1 = I(abs(max(maximum(xgrid), 0)))
        z1 = I(abs(max(maximum(zgrid), 0)))

        return Stencil(_xgrid, _zgrid, _xcoefs, _zcoefs, x0, z0, x1, z1)
    end

    function Stencil(xorder, zorder, Δx, Δz; device=CPU())
        @assert iseven(xorder) && iseven(zorder) "Called Stencil constructor is only defined for even orders"
        xpad = xorder ÷ 2
        zpad = zorder ÷ 2
        xgrid = filter(!iszero, -xpad:xpad)
        zgrid = filter(!iszero, -zpad:zpad)

        Stencil(xgrid, zgrid, 0, 0, Δx, Δz, device=device)
    end

    function Stencil(order, h; device=CPU())
        Stencil(order, order, h, h, device=device)
    end
end


@kernel inbounds = true unsafe_indices = true function _ddx_kernel_padded!(du, u, grid, coefs, x0, z0)
    I = @index(Global, NTuple)
    nx = I[1] + x0
    nz = I[2] + z0

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        val += c * u[nx+g, nz]
    end
    du[nx, nz] = val
end

@kernel inbounds = true unsafe_indices = true function _ddz_kernel_padded!(du, u, grid, coefs, x0, z0)
    I = @index(Global, NTuple)
    nx = I[1] + x0
    nz = I[2] + z0

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        val += c * u[nx, nz+g]
    end
    du[nx, nz] = val
end

@kernel function _ddx_kernel_periodic_left!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nz = I[2]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nx = I[1] + g
        nx += (nx < 1) ? Nx : 0

        val += c * u[nx, nz]
    end
    du[I] = val
end

@kernel function _ddx_kernel_periodic_right!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nz = I[2]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nx = Nx - I[1] + 1 + g
        nx += (nx > Nx) ? -Nx : 0
        
        val += c * u[nx, nz]
    end
    du[I] = val
end

@kernel function _ddx_kernel_periodic_top!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nx = I[1]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nz = I[2] + g
        nz += (nz < 1) ? Nz : 0

        val += c * u[nx, nz]
    end
    du[I] = val
end

@kernel function _ddx_kernel_periodic_bottom!(du, u, grid, coefs, Nx, Nz)
    I = @index(Global, Cartesian)
    nx = I[1]

    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]

        nz = Nz - I[2] + 1 + g
        nz += (nz > Nz) ? -Nz : 0
        
        val += c * u[nx, nz]
    end
    du[I] = val
end

function ddx!(du, u, fdm::Stencil)
    device = get_backend(du)
    kernel! = _ddx_kernel_padded!(device, 64)
    Nx, Nz = size(du)
    ndrange = (Nx - fdm.x0 - fdm.x1, Nz - fdm.z0 - fdm.z1)
    kernel!(du, u, fdm.xgrid, fdm.xcoefs, fdm.x0, fdm.z0; ndrange=ndrange)
    return nothing
end
function ddz!(du, u, fdm::Stencil)
    device = get_backend(du)
    kernel! = _ddz_kernel_padded!(device, 64)
    Nx, Nz = size(du)
    ndrange = (Nx - fdm.x0 - fdm.x1, Nz - fdm.z0 - fdm.z1)
    kernel!(du, u, fdm.zgrid, fdm.zcoefs, fdm.x0, fdm.z0; ndrange=ndrange)
    return nothing
end


function ddx_synced!(du, u, fdm::Stencil)
    device = get_backend(du)
    ddx!(du, u, fdm)
    KA.synchronize(device)
    return nothing
end
function ddz_synced!(du, u, fdm::Stencil)
    device = get_backend(du)
    ddz!(du, u, fdm)
    KA.synchronize(device)
    return nothing
end