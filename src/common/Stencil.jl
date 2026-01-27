
"""
Generate stencil coefficients for a given order and spacing.
Might be ill-conditioned for very high orders.
Using Rational numbers give larger stability

https://discourse.julialang.org/t/generating-finite-difference-stencils/85876/5
"""
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
    Stencil(xorder, zorder, Δx, Δz, Val(:staggered_minus); device=CPU())
    Stencil(xorder, zorder, Δx, Δz, Val(:staggered_plus); device=CPU())
    Stencil(order, h, Val(:staggered_minus); device=CPU())
    Stencil(order, h, Val(:staggered_plus); device=CPU())

High-order finite difference stencil for computing spatial derivatives.

# Constructors
- `Stencil(order, h; device=CPU())`: Creates isotropic stencil with given order and spacing
- `Stencil(xorder, zorder, Δx, Δz; device=CPU())`: Creates anisotropic stencil with different orders and spacings
- `Stencil(xgrid, zgrid, x0, z0, Δx, Δz; device=CPU())`: Creates stencil from custom grid points

- `Stencil(xorder, zorder, Δx, Δz, Val(:staggered_minus); device=CPU())`: Creates forward staggered grid stencil for velocity-stress formulation, i.e. [-1, 0, 1, 2] => [-3/2, -1/2, 1/2, 3/2]
- `Stencil(xorder, zorder, Δx, Δz, Val(:staggered_plus); device=CPU())`: Creates backward staggered grid stencil for velocity-stress formulation, i.e. [-2, -1, 0, 1] => [-3/2, -1/2, 1/2, 3/2]
- `Stencil(order, h, Val(:staggered_minus); device=CPU())`: Convenience constructor when x- and z-order and step sizes are the same
- `Stencil(order, h, Val(:staggered_plus); device=CPU())`: Convenience constructor when x- and z-order and step sizes are the same

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

# 4th-order staggered grid stencil for velocity-stress formulation
stencil⁺ = Stencil(4, 1.0, stagger=:plus)
stencil⁻ = Stencil(4, 1.0, stagger=:minus)

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

    function Stencil(xorder, zorder, Δx, Δz; stagger=:none, device=CPU())
        @assert iseven(xorder) && iseven(zorder) "Staggered stencil requires even orders"
        xpad = xorder ÷ 2
        zpad = zorder ÷ 2

        if stagger == :minus
            xgrid = collect(-xpad+1:xpad)
            zgrid = collect(-zpad+1:zpad)

            # Half-grid positions relative to i+1/2
            # e.g., order 4: indices [-1,0,1,2] -> positions [-3/2, -1/2, 1/2, 3/2]
            xloc = [i - 1//2 for i in xgrid]
            zloc = [i - 1//2 for i in zgrid]
        elseif stagger == :plus
            xgrid = collect(-xpad:xpad-1)
            zgrid = collect(-zpad:zpad-1)

            # Half-grid positions relative to evaluation point i
            # e.g., order 4: indices [-2,-1,0,1] -> positions [-3/2, -1/2, 1/2, 3/2]
            # (accessing staggered field v[j] at position j+1/2)
            xloc = [i + 1//2 for i in xgrid]
            zloc = [i + 1//2 for i in zgrid]
        else
            xgrid = xloc = filter(!iszero, -xpad:xpad)
            zgrid = zloc = filter(!iszero, -zpad:zpad)
        end

        I = preferred_int(device)
        F = preferred_float(device)
        _xgrid = allocate(device, I, size(xgrid))
        _zgrid = allocate(device, I, size(zgrid))
        _xcoefs = similar(_xgrid, F)
        _zcoefs = similar(_zgrid, F)

        xcoefs = _stencil(xloc, 0, 1) ./ Δx
        zcoefs = _stencil(zloc, 0, 1) ./ Δz

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

    function Stencil(order, h; stagger=:none, device=CPU()) 
        Stencil(order, order, h, h; stagger=stagger, device=device)
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