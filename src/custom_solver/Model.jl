using KernelAbstractions
using FiniteDifferences

struct Solver
    vx
    vz
    sxx
    szz
    sxz
end

struct Model
    ε
    δ
    γ
end

@inline function ddx!(u, m, n, grid, coefs)
    val = zero(eltype(du))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        val += c * u[m + g, n]
    end
    return val
end
@inline function ddz!(u, m, n, grid, coefs)
    val = zero(eltype(u))
    for i in eachindex(grid)
        c = coefs[i]
        g = grid[i]
        val += c * u[m, n + g]
    end
    return val
end

fdm_o8 = central_fdm(8,1)

ddx_o8! = (u,m,n) -> ddx!(u,m,n,fdm_o8.grid,fdm_o8.coefs)
ddz_o8! = (u,m,n) -> ddx!(u,m,n,fdm_o8.grid,fdm_o8.coefs)
function stagger_grid(u)
    # TODO: Implement
    error("Grid staggering not implemented.")
end


"""
    timestep_strain_interior!(sxx, szz, sxz, vx, vz, C, pad, grid, coefs, Δt)

Handle the leapfrog step for interior strain following Fichtner (2012).
    Boundary update must be called separately. If the size of the fields are `Nx` and `Nz`

# Arguments
- `sxx`: Matrix(Nx,Nz) containing xx-component of the Cauchy stress tensor.
- `szz`: Matrix(Nx,Nz) containing zz-component of the Cauchy stress tensor.
- `sxz`: Matrix(Nx,Nz) containing xz-component of the Cauchy stress tensor.
- `vx`: Matrix(Nx,Nz) containing x-component of the velocity vector.
- `vz`: Matrix(Nx,Nz) containing z-component of the velocity vector.
- `C`: Array(6,Nx,Nz) of elastic anisotropic parameters, 
    Voigt notation in the following order: 
    `C[1,:,:]` = ``C_{11}(x,z)`` 
    `C[2,:,:]` = ``C_{13}(x,z)`` 
    `C[3,:,:]` = ``C_{15}(x,z)`` 
    `C[4,:,:]` = ``C_{33}(x,z)`` 
    `C[5,:,:]` = ``C_{35}(x,z)`` 
    `C[6,:,:]` = ``C_{55}(x,z)``
- `pad`: Integer describing interior field padding based on finite difference stencil.
- `grid`: Vector of integers for the finite difference scheme.
- `coefs`: Vector of coefficients for the finite difference scheme.
- `Δt`: Time step.

# Returns
- `nothing`: In-place update of the stress tensors sxx, szz, sxz.
"""
@kernel function timestep_strain_interior!(sxx, szz, sxz, vx, vz, C, pad, grid, coefs, Δt, zdiff = (u, m, n))
    # Handle the interior leapfrog step for strain following Fichtner (2012)
    
    I = @index(Global, Cartesian)
    m = I[1] + pad
    n = I[2] + pad

    # TODO: Interpolation of coefficients
    c11 = C[1, m, n]
    c13 = C[2, m, n]
    c15 = C[3, m, n]
    c33 = C[4, m, n]
    c35 = C[5, m, n]
    c55 = C[6, m, n]

    # 1. Compute ̇ε(t) from v(t)
    ∂vx_∂x = ddx!(vx, m, n, grid, coefs)
    ∂vx_∂z = ddz!(vx, m, n, grid, coefs)

    ∂vz_∂x = ddx!(vz, m, n, grid, coefs)
    ∂vz_∂z = ddz!(vz, m, n, grid, coefs)

    # 2. Compute ̇s(t) from ̇ε(t)
    ∂sxx_∂t = c11 * ∂vx_∂x + c13 * ∂vz_∂z + c15*(∂vx_∂z + ∂vz_∂x)
    ∂szz_∂t = c13 * ∂vx_∂x + c33 * ∂vz_∂z + c35*(∂vx_∂z + ∂vz_∂x)
    ∂sxz_∂t = c15 * ∂vx_∂x + c35 * ∂vz_∂z + c55*(∂vx_∂z + ∂vz_∂x)

    # 3. Time step s(t + ½Δt) from ̇s(t)
    sxx[m, n] = sxx[m, n] + ∂sxx_∂t * Δt
    szz[m, n] = szz[m, n] + ∂szz_∂t * Δt
    sxz[m, n] = sxz[m, n] + ∂sxz_∂t * Δt

    return nothing # in-place update
end

@kernel function timestep_velocity_interior!(vx, vz, sxx, szz, sxz, αxs, αzs, fxs, fzs, pad, grid, coefs, Δt)
    # Handle the interior leapfrog timestep for velocity
    # Following Fichtner (2012)

    I = @index(Global, Cartesian)
    m = I[1] + pad
    n = I[2] + pad

    # buoyancy and forces must be interpolated before they are passed into the function
    αx = αxs[m, n]
    αz = αzs[m, n]
    fx = fxs[m, n]
    fz = fzs[m, n]

    # 4. Compute ∇⋅s(t + ½Δt)
    ∂sxx_∂x = ddx!(sxx, m, n, grid, coefs)
    ∂szz_∂z = ddz!(szz, m, n, grid, coefs)

    ∂sxz_∂x = ddx!(sxz, m, n, grid, coefs)
    ∂szx_∂z = ∂sxz_∂x # Symmetry

    # Compute ̇v from momentum balance
    ∂vx_∂t = αx * (∂sxx_∂x + ∂szx_∂z + fx)
    ∂vz_∂t = αz * (∂sxz_∂x + ∂szz_∂z + fz)

    # 5. Compute v(t + Δt) from ̇v(t)
    vx[m, n] = vx[m, n] + ∂vx_∂t * Δt
    vz[m, n] = vz[m, n] + ∂vz_∂t * Δt

    return nothing # in-place update
end

function timestep_velocity(model)
    device = get_backend(model.vx)
    # kernel = timestep_velocity_interior!(model.vx,model.vz,model.sxx)
end


function test_timestep()

    try 
        Zygote.gradient()
    catch e
        return false
    end

    return true

end






