@inline function ddxp(i, j, n, u, D, L)
    val = zero(eltype(u))
    for ℓ in 1:L
        val += D[ℓ] * (u[i+ℓ, j, n] - u[i-ℓ+1, j, n])
    end
    return val
end
@inline function ddxm(i, j, n, u, D, L)
    val = zero(eltype(u))
    for ℓ in 1:L
        val += D[ℓ] * (u[i+ℓ-1, j, n] - u[i-ℓ, j, n])
    end
    return val
end

@inline function ddzp(i, j, n, u, D, L)
    val = zero(eltype(u))
    for ℓ in 1:L
        val += D[ℓ] * (u[i, j+ℓ, n] - u[i, j-ℓ+1, n])
    end
    return val
end

@inline function ddzm(i, j, n, u, D, L)
    val = zero(eltype(u))
    for ℓ in 1:L
        val += D[ℓ] * (u[i, j+ℓ-1, n] - u[i, j-ℓ, n])
    end
    return val
end

@kernel function _vel_iso_kernel!(dv, s, ρs, D, DL, pad)
    I = @index(Global, NTuple)
    i, j = I[1]+pad, I[2]+pad
    
    # Calculate b(i+½, j) and b(i, j+½)
    # buoyancy factors at half integer positions
    bi = 2.0 / (ρs[i,j] + ρs[i+1,j])
    bj = 2.0 / (ρs[i,j] + ρs[i,j+1])

    dxsxx = ddxp(i, j, 1, s, D, DL)
    dzszz = ddzp(i, j, 2, s, D, DL)
    dxszx = ddxm(i, j, 3, s, D, DL)
    dzsxz = ddzm(i, j, 3, s, D, DL) # σxz=σzx

    dv[i, j, 1] = bi * (dxsxx + dzsxz)
    dv[i, j, 2] = bj * (dzszz + dxszx)
    return nothing
end

@kernel function _stress_iso_kernel!(ds, v, λs, μs, D, DL, pad)
    I = @index(Global, NTuple)
    i, j = I[1]+pad, I[2]+pad

    λ = λs[i,j]
    # Reference: μ(i,j)
    μ = μs[i,j]
    # At the σxz node: μ(i+½, j+½)
    μ_half = 0.25*(μ + μs[i+1,j] + μs[i,j+1] + μs[i+1,j+1])

    dxvx = ddxm(i, j, 1, v, D, DL)
    dzvz = ddzm(i, j, 2, v, D, DL)
    dzvx = ddzp(i, j, 1, v, D, DL)
    dxvz = ddxp(i, j, 2, v, D, DL)

    ds[i, j, 1] = λ * (dxvx + dzvz) + 2μ * dxvx
    ds[i, j, 2] = λ * (dxvx + dzvz) + 2μ * dzvz
    ds[i, j, 3] = μ_half * (dzvx + dxvz)
    return nothing
end

@kernel function _stress_1Q_iso_kernel!(ds, v, λs, μs, τs, τn, D, DL, pad)
    I = @index(Global, NTuple)
    i,j = I[1]+pad, I[2]+pad

    λ = λs[i,j]
    μ = μs[i,j]
    τ = τs[i,j]
    N = length(τn)
    
    εxx = ddxm(i, j, 1, v, D, DL)
    εzz = ddzm(i, j, 2, v, D, DL)
    εxz = ddzp(i, j, 1, v, D, DL) + ddxp(i, j, 2, v, D, DL)

    λ0 = λ * (1 + τ)
    μ0 = μ * (1 + τ)
    M11 = τ * sum(M[i,j,n] for n in 1:N)
    M13 = λ * τ * sum(M[i,j,n] for n in 1:N)
    M55 = μ * τ * sum(M[i,j,n] for n in 1:N)

    ds[i,j,1] = (λ0 + 2μ0)*εxx + λ0*εzz +
                (λ + 2μ)*τ*M11 + M13
    ds[i,j,2] = λ0*εxx + (λ0 + 2μ0)*εzz + M13 + M11
    ds[i,j,3] = μ0*εxz + M55
    return nothing
end

@kernel function _stress_free_surface_kernel!(szz, sxz, depth)
    i, j = @index(Global, NTuple)
    for d in 1:depth
        szz[i, j+d] = -sxz[i, j]
    end
    return nothing
end

@kernel function _velocity_free_surface_kernel!()
    
end