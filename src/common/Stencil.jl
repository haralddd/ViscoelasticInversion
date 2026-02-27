
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

# s4 = _stencil([-1//1, (-1//2:1//1:5//2)...], 0//1, 1)

# Make a new stencil struct more suitable for the visco-kernels

struct StaggeredStencil{T}
    D::Vector{T}
    DL::Int

    function StaggeredStencil(DL::Int, device=CPU())
        g1 = (DL+1)//2
        coeffs = _stencil(-g1:g1, 0//1, 1)

        T = preferred_float(device)
        D = allocate(device, T, length(coeffs))
        copyto!(D, Vector{T}(coeffs))
        return new{T}(D, DL)
    end
end


"""
 H-AFDA Free surface stencil without stress imaging, described by Kristek 2002
 https://doi.org/10.1023/A:1019866422821 

 Grid layout (H-formulation, surface at half-grid above first integer point):
   z = 0      : free surface (τzz=0, τxz=0)
   z = h/2    : vz, τxz positions
   z = h      : vx, τxx, τzz positions  
   z = 3h/2   : vz, τxz positions
   ...

 Formulas (Table 4):
   D1: derivative at z=0, with f(0)=0 (formula #1) - for τxz,z at surface
   D2: derivative at z=h/2 (formula #2) - for u,z, v,z, τzz,z at half-grid
   D4: derivative at z=h, with f(0)=0 (formula #4) - for τxz,z at first integer point
"""
struct FreeSurfaceStencil{T}
    D1::Vector{T}  # Formula #1: f'(0) with f(0)=0, uses points at h/2, 3h/2, ...
    D2::Vector{T}  # Formula #2: f'(h/2), uses points at 0, h, 2h, ...
    D3::Vector{T}  # Formula #3: f'(h) Hermitian, uses f'(0) + points at h/2, 3h/2, ...
    D3_deriv::T    # Formula #3: coefficient for f'(0) term
    D4::Vector{T}  # Formula #4: f'(h) with f(0)=0, uses points at h/2, 3h/2, ... (explicit alternative to #3)
    L::Int         # Half-length of standard stencil (order = 2L)

    function FreeSurfaceStencil(order::Int, device=CPU())
        L = order ÷ 2
        
        # Grid points for stencils (in units of h)
        # D1: f'(0) using f(0)=0, sample at half-integers h/2, 3h/2, ..., (2L-1)h/2
        pts_half = [(2k-1)//2 for k in 1:order]  # [1/2, 3/2, 5/2, ...]
        
        # D2: f'(h/2) using integer points 0, h, 2h, ..., Lh
        pts_int = [k//1 for k in 0:order]  # [0, 1, 2, ...]
        
        # D3: Hermitian stencil for f'(h) using f'(0) and half-grid points
        # We solve for coefficients such that f'(h) ≈ α·f'(0) + Σ βₖ·f(kh/2)
        # This requires solving an augmented system
        pts_D3 = [0//1, pts_half...]  # Include 0 as "derivative point"
        
        # Compute stencil coefficients
        s1_full = _stencil(pts_half, 0//1, 1)      # f'(0) from half-grid points
        s2_full = _stencil(pts_int, 1//2, 1)       # f'(h/2) from integer points
        s4_full = _stencil(pts_half, 1//1, 1)      # f'(h) explicit (alternative to #3)

        s1 = s1_full[2:end] # Omit evaluation on the free surface f(0) = 0
        s2 = s2_full[2:end]
        s4 = s4_full[2:end]
        
        # Formula #3: Hermitian stencil
        # f'(h) = α·h·f'(0) + Σ βₖ·f((2k-1)h/2)  for k=1,...,order
        # The system includes: derivative info at 0, function values at half-grid
        # Augmented Vandermonde: row for derivative at 0 contributes to accuracy
        s3_full = _stencil_hermitian(pts_half, 0//1, 1//1, 1)  # Custom Hermitian stencil
        s3_deriv = s3_full[1]     # Coefficient for f'(0) term (multiplied by h)
        s3 = s3_full[2:end]       # Coefficients for function values
        
        T = preferred_float(device)
        D1 = allocate(device, T, length(s1))
        D2 = allocate(device, T, length(s2))
        D3 = allocate(device, T, length(s3))
        D4 = allocate(device, T, length(s4))
        
        copyto!(D1, Vector{T}(s1))
        copyto!(D2, Vector{T}(s2))
        copyto!(D3, Vector{T}(s3))
        copyto!(D4, Vector{T}(s4))
        
        return new{T}(D1, D2, D3, T(s3_deriv), D4, L)
    end
end

"""
    _stencil_hermitian(x, x_deriv, x₀, m)

Hermitian stencil: approximate f^(m)(x₀) using f'(x_deriv) and function values at x.
Returns [coef_for_derivative, coefs_for_function_values...]

# Arguments
- `x`: Vector of points where function values are known
- `x_deriv`: Single point where derivative is known  
- `x₀`: Point where we want to approximate the derivative
- `m`: Order of derivative to approximate (1 = first derivative)
"""
function _stencil_hermitian(x::AbstractVector{<:Real}, x_deriv::Real, x₀::Real, m::Integer)
    return _stencil_hermitian(x, [x_deriv], [1], x₀, m)
end

"""
    _stencil_hermitian(x_func, x_derivs, deriv_orders, x₀, m)

Full Hermitian stencil: approximate f^(m)(x₀) using:
- Function values f(x) at points in `x_func`
- Derivatives f^(k)(x) at points in `x_derivs` with orders in `deriv_orders`

Returns [coefs_for_derivatives..., coefs_for_function_values...]

# Arguments
- `x_func`: Vector of points where function values are known
- `x_derivs`: Vector of points where derivatives are known
- `deriv_orders`: Vector of derivative orders at each point in x_derivs (e.g., [1,1] for two first derivatives)
- `x₀`: Point where we want to approximate the derivative
- `m`: Order of derivative to approximate

# Example
```julia
# Approximate f'(1) using f(1/2), f(3/2), f(5/2), f(7/2) and f'(0)
_stencil_hermitian([1//2, 3//2, 5//2, 7//2], [0//1], [1], 1//1, 1)

# Approximate f'(1) using f(1/2), f(3/2) and both f'(0) and f''(0)
_stencil_hermitian([1//2, 3//2], [0//1, 0//1], [1, 2], 1//1, 1)

# Full Hermitian: use f and f' at multiple points
_stencil_hermitian([0//1, 1//1], [0//1, 1//1], [1, 1], 1//2, 1)  # f'(1/2) from f(0),f(1),f'(0),f'(1)
```
"""
function _stencil_hermitian(x_func::AbstractVector{<:Real}, x_derivs::AbstractVector{<:Real}, 
                            deriv_orders::AbstractVector{<:Integer}, x₀::Real, m::Integer)
    n_func = length(x_func)
    n_deriv = length(x_derivs)
    n_total = n_func + n_deriv
    
    # Total degrees of freedom determines max polynomial order we can match
    ℓ_max = n_total - 1
    
    # Build augmented Vandermonde matrix
    A = zeros(Rational{BigInt}, n_total, n_total)
    
    # Columns 1:n_deriv: derivative values
    # f^(k)(x_d) = Σ_{ℓ=k}^{∞} f^(ℓ)(x₀) (x_d - x₀)^(ℓ-k) / (ℓ-k)!
    for (col, (xd, k)) in enumerate(zip(x_derivs, deriv_orders))
        for row in 1:n_total
            ℓ = row - 1
            if ℓ >= k
                A[row, col] = (big(xd) - big(x₀))^(ℓ - k) // factorial(big(ℓ - k))
            end
        end
    end
    
    # Columns (n_deriv+1):n_total: function values
    # f(x) = Σ_{ℓ=0}^{∞} f^(ℓ)(x₀) (x - x₀)^ℓ / ℓ!
    for (idx, xk) in enumerate(x_func)
        col = n_deriv + idx
        for row in 1:n_total
            ℓ = row - 1
            A[row, col] = (big(xk) - big(x₀))^ℓ // factorial(big(ℓ))
        end
    end
    
    # Right-hand side: we want the m-th derivative at x₀
    b = zeros(Rational{BigInt}, n_total)
    b[m+1] = 1
    
    return A \ b
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