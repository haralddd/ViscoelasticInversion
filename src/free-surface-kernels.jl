include("diff-funcs.jl")



# =============================================================================
# H-AFDA Free Surface Kernels (Kristek 2002)
# =============================================================================
# Grid layout (j index, surface at j=1):
#   j=1: vx, τxx, τzz (integer grid) - FREE SURFACE
#   j=1: vz, τxz (half-grid, at j+1/2)
#   j=2: vx, τxx, τzz (integer grid)
#   ...

# Velocity update at free surface (j=1): vx
# Uses formula #1 for τxz,z with τxz(0)=0
@kernel function _vel_freesurf_j1!(v, s, ρs, D1, Dx, dt)
    pad = length(Dx)
    i = @index(Global) + pad
    j = 1  # Surface
    
    bi = 2.0 / (ρs[i, j] + ρs[i+1, j])
    
    # τxx,x uses standard stencil, τzx,z uses formula #1 (one-sided)
    dvx = bi * (ddxp(i, j, 1, s, Dx) + ddzm_D1(i, j, 3, s, D1))
    
    v[i, j, 1] += dt * dvx
end

# Velocity update at j=1 half-grid: vz (at j+1/2)
# Uses formula #2 for τzz,z
@kernel function _vel_freesurf_vz!(v, s, ρs, D2, Dx, dt)
    pad = length(Dx)
    i = @index(Global) + pad
    j = 1
    
    bj = 2.0 / (ρs[i, j] + ρs[i, j+1])
    
    # τzz,z uses formula #2 (one-sided from integer points)
    # τxz,x uses standard stencil
    dvz = bj * (ddzp_D2(i, j, 2, s, D2) + ddxm(i, j, 3, s, Dx, DL))
    
    v[i, j, 2] += dt * dvz
end

# Velocity update at j=2: vx
# Uses formula #4 for τxz,z with τxz(0)=0
@kernel function _vel_freesurf_j2!(v, s, ρs, D4, Dx, dt)
    pad = length(Dx)
    i = @index(Global) + pad
    j = 2
    
    bi = 2.0 / (ρs[i, j] + ρs[i+1, j])
    
    # τxz,z uses formula #4
    dvx = bi * (ddxp(i, j, 1, s, Dx, DL) + ddzm_D4(i, j, 3, s, D4))
    
    v[i, j, 1] += dt * dvx
end

# Stress update at free surface (j=1): τxx, τzz
# τzz(0) = 0 (boundary condition)
# τxx uses: w,z replaced by (u,x + v,y) due to τzz=0 condition
# In 2D: τzz=0 implies (λ+2μ)w,z + λ(u,x) = 0, so w,z = -λ/(λ+2μ) * u,x
@kernel function _stress_freesurf_j1!(s, M, v, λs, μs, τs, τns, D2, Dx, dt)
    pad = length(Dx)
    i = @index(Global) + pad
    j = 1  # Surface
    N = length(τns)
    
    # Strain rates
    εxx = ddxm(i, j, 1, v, Dx)
    # εzz uses formula #2 for vz,z (one-sided)
    εzz = ddzp_D2(i, j, 2, v, D2)
    # εxz uses formula #2 for vx,z
    εxz = 0.5 * (ddzp_D2(i, j, 1, v, D2) + ddxp(i, j, 2, v, Dx, DL))
    
    # Material properties
    λr = λs[i, j]
    μr = μs[i, j]
    τ = τs[i, j]
    
    πr = λr + 2μr
    
    # Apply free surface condition: τzz = 0
    # This means we need to compute τxx with the constraint
    # τzz = (λ+2μ)εzz + λεxx = 0  =>  εzz_eff = -λ/(λ+2μ) * εxx
    # But for τxx = (λ+2μ)εxx + λεzz, with τzz=0:
    # τxx = (λ+2μ)εxx + λ*(-λ/(λ+2μ)*εxx) = (λ+2μ - λ²/(λ+2μ))εxx
    #     = ((λ+2μ)² - λ²)/(λ+2μ) * εxx = (4μλ + 4μ²)/(λ+2μ) * εxx
    #     = 4μ(λ+μ)/(λ+2μ) * εxx
    # Or simply: τxx = 2μ*(2εxx) for incompressible, but general case:
    
    # For viscoelastic, use relaxed moduli
    π1 = πr * τ
    π0 = π1 + πr
    λ1 = λr * τ
    λ0 = λ1 + λr
    
    ΣMxx = sum(M[i, j, n, 1] for n in axes(M, 3))
    ΣMzz = sum(M[i, j, n, 2] for n in axes(M, 3))
    
    # Standard constitutive relation for τxx (τzz is set to 0)
    dsxx = π0 * εxx + λ0 * εzz + π1 * ΣMxx + λ1 * ΣMzz
    
    s[i, j, 1] += dt * dsxx
    s[i, j, 2] = 0.0  # τzz = 0 at free surface
    # τxz at j=1 position is actually at j+1/2, handled separately
    
    # Memory variable update
    for n in axes(M, 3)
        τn = τns[n]
        a1 = -1.0 / (N * τn)
        a2 = -1.0 / τn
        
        M[i, j, n, 1] += dt * (a1 * εxx + a2 * M[i, j, n, 1])
        M[i, j, n, 2] = 0.0  # Consistent with τzz=0
    end
end

# Stress update at j=1 half-grid: τxz (at j+1/2)
# Uses formula #2 for u,z
@kernel function _stress_freesurf_txz!(s, v, μs, τs, τns, M, D2, Dx, DL, dt)
    i = @index(Global) + DL
    j = 1
    N = length(τns)
    
    # εxz uses formula #2 for vx,z (one-sided)
    εxz = 0.5 * (ddzp_D2(i, j, 1, v, D2) + ddxp(i, j, 2, v, Dx, DL))
    
    μr = μs[i, j]
    μrxz = 0.25 * (μr + μs[i+1, j] + μs[i, j+1] + μs[i+1, j+1])
    τ = τs[i, j]
    
    μ1xz = μrxz * τ
    μ0xz = μ1xz + μrxz
    
    ΣMxz = sum(M[i, j, n, 3] for n in axes(M, 3))
    
    dsxz = μ0xz * εxz + μ1xz * ΣMxz
    s[i, j, 3] += dt * dsxz
    
    for n in axes(M, 3)
        τn = τns[n]
        a1 = -1.0 / (N * τn)
        a2 = -1.0 / τn
        M[i, j, n, 3] += dt * (a1 * εxz + a2 * M[i, j, n, 3])
    end
end

# Stress update at j=2 (z=h): τxx, τzz using Formula #3 (Hermitian)
# Uses w,z(0) derived from τzz(0)=0 condition:
#   τzz(0) = (λ+2μ)w,z(0) + λ*u,x(0) = 0
#   => w,z(0) = -λ/(λ+2μ) * u,x(0)
@kernel function _stress_freesurf_j2!(s, M, v, λs, μs, τs, τns, D3, D3_deriv, Dx, DL, dt)
    i = @index(Global) + DL
    j = 2  # First interior integer point
    N = length(τns)
    
    # First compute w,z(0) from the free surface condition
    # Need u,x at j=1 (surface)
    εxx_surf = ddxm(i, 1, 1, v, Dx, DL)
    λr_surf = λs[i, 1]
    μr_surf = μs[i, 1]
    πr_surf = λr_surf + 2μr_surf
    
    # From τzz(0) = 0: w,z(0) = -λ/(λ+2μ) * u,x(0)
    vz_z_at_surface = -λr_surf / πr_surf * εxx_surf
    
    # Strain rates at j=2
    εxx = ddxm(i, j, 1, v, Dx, DL)
    # εzz uses formula #3 (Hermitian) with vz,z(0) computed above
    εzz = ddzp_D3(i, j, 2, v, D3, D3_deriv, vz_z_at_surface)
    # εxz: vx,z uses formula #3, vz,x uses standard stencil
    vx_z = ddzp_D3(i, j, 1, v, D3, D3_deriv, zero(eltype(v)))  # vx,z(0) ≈ 0 at free surface
    εxz = 0.5 * (vx_z + ddxp(i, j, 2, v, Dx, DL))
    
    # Material properties
    λr = λs[i, j]
    μr = μs[i, j]
    τ = τs[i, j]
    
    πr = λr + 2μr
    π1 = πr * τ
    π0 = π1 + πr
    λ1 = λr * τ
    λ0 = λ1 + λr
    
    μrxz = 0.25 * (μr + μs[i+1, j] + μs[i, j+1] + μs[i+1, j+1])
    μ1xz = μrxz * τ
    μ0xz = μ1xz + μrxz
    
    ΣMxx = sum(M[i, j, n, 1] for n in axes(M, 3))
    ΣMzz = sum(M[i, j, n, 2] for n in axes(M, 3))
    ΣMxz = sum(M[i, j, n, 3] for n in axes(M, 3))
    
    dsxx = π0 * εxx + λ0 * εzz + π1 * ΣMxx + λ1 * ΣMzz
    dszz = λ0 * εxx + π0 * εzz + λ1 * ΣMxx + π1 * ΣMzz
    dsxz = μ0xz * εxz + μ1xz * ΣMxz
    
    s[i, j, 1] += dt * dsxx
    s[i, j, 2] += dt * dszz
    s[i, j, 3] += dt * dsxz
    
    # Memory variable update
    for n in axes(M, 3)
        τn = τns[n]
        a1 = -1.0 / (N * τn)
        a2 = -1.0 / τn
        
        M[i, j, n, 1] += dt * (a1 * εxx + a2 * M[i, j, n, 1])
        M[i, j, n, 2] += dt * (a1 * εzz + a2 * M[i, j, n, 2])
        M[i, j, n, 3] += dt * (a1 * εxz + a2 * M[i, j, n, 3])
    end
end