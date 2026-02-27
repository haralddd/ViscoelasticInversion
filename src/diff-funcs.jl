
# Staggered finite difference operators D⁺ and D⁻.
function ddxp(i, j, n, u, D)
    val = zero(eltype(u))
    for ℓ in eachindex(D)
        val += D[ℓ] * (u[i+ℓ, j, n] - u[i-ℓ+1, j, n])
    end
    return val
end
function ddxm(i, j, n, u, D)
    val = zero(eltype(u))
    for ℓ in eachindex(D)
        val += D[ℓ] * (u[i+ℓ-1, j, n] - u[i-ℓ, j, n])
    end
    return val
end

function ddzp(i, j, n, u, D)
    val = zero(eltype(u))
    for ℓ in eachindex(D)
        val += D[ℓ] * (u[i, j+ℓ, n] - u[i, j-ℓ+1, n])
    end
    return val
end

function ddzm(i, j, n, u, D)
    val = zero(eltype(u))
    for ℓ in eachindex(D)
        val += D[ℓ] * (u[i, j+ℓ-1, n] - u[i, j-ℓ, n])
    end
    return val
end

# H-AFDA free surface derivative functions
# These use one-sided stencils for points near the free surface at j=1 (z=0)

# Formula #1: f'(0) with f(0)=0, using half-grid points at j=1,2,3,...
# Used for: τzx,z at surface (vx update at z=0)
function ddzm_D1(i, n, u, coefs)
    for k in eachindex(coefs)
        # First coefficient in the full FD stencil is dropped in constructor
        # due to the boundary condition f(0)=0
        val += coefs[k] * u[i, k, n]
    end
    return val
end

# Formula #2: f'(h/2), using integer points at j=1,2,...
# Used for: dzvx, dzvz in τxz(h/2) update position, τzz,z at vz position
function ddzp_D2(i, j, n, u, coefs, c0, c1, u0, u1)
    val = c0 * u0
    val += c1 * u1
    for k in eachindex(coefs)
        val += coefs[k] * u[i, j+k-1, n]  # points at j, j+1, j+2, ...
    end
    return val
end

# Formula #3: f'(h) Hermitian, using f'(0) + half-grid points
# Used for: w,z at stress update (j=2), requires pre-computed f'(0)
function ddzp_D3(i, j, n, u, coefs, c0, du0)
    val = c0 * du0  # Contribution from derivative at surface
    for k in eachindex(coefs)
        val += coefs[k] * u[i, j+k-1, n]  # points at j, j+1, j+2, ...
    end
    return val
end

# Formula #4: f'(h) with f(0)=0, using half-grid points
# Used for: τxz,z at first interior point (vx update at j=2)
function ddzm_D4(i, j, n, u, coefs)
    val = zero(eltype(u))
    for k in eachindex(coefs)
        val += coefs[k] * u[i, j+k-1, n]  # points at j, j+1, j+2, ...
    end
    return val
end