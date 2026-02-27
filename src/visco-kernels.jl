include("diff-funcs.jl")

@kernel function _vel_1Q!(v, s, ρs, D, dt)
    I = @index(Global, NTuple)
    pad = length(D)
    i, j = I[1] + pad, I[2] + pad

    # Calculate b(i+½, j) and b(i, j+½)
    # buoyancy factors at half integer positions
    bi = 2.0 / (ρs[i, j] + ρs[i+1, j])
    bj = 2.0 / (ρs[i, j] + ρs[i, j+1])

    dvx = bi * (
        ddxm(i, j, 1, s, D) + ddzp(i, j, 3, s, D))
    dvz = bj * (
        ddzm(i, j, 2, s, D) + ddxp(i, j, 3, s, D))

    v[i, j, 1] += dt * dvx
    v[i, j, 2] += dt * dvz
end

function _ds_1Q(M, i, j, εxx, εzz, εxz, λs, μs, τs)
    λr = λs[i, j]
    μr = μs[i, j]
    μrxz = 0.25 * (μr + μs[i+1, j] + μs[i, j+1] + μs[i+1, j+1])
    τ = τs[i, j]

    πr = (λr + 2μr)
    π1 = πr * τ
    π0 = π1 + πr

    λ1 = λr * τ
    λ0 = λ1 + λr
    
    μ1xz = μrxz * τ
    μ0xz = μ1xz + μrxz

    ΣMxx = sum(M[i, j, n, 1] for n in axes(M, 3))
    ΣMzz = sum(M[i, j, n, 2] for n in axes(M, 3))
    ΣMxz = sum(M[i, j, n, 3] for n in axes(M, 3))

    dsxx = π0 * εxx + λ0 * εzz + π1 * ΣMxx + λ1 * ΣMzz
    dszz = λ0 * εxx + π0 * εzz + λ1 * ΣMzz + π1 * ΣMxx
    dsxz = μ0xz * εxz + μ1xz * ΣMxz
    return dsxx, dszz, dsxz
end


function _dM_1Q(Mxxn, Mzzn, Mxzn, εxx, εzz, εxz, τn, N)
    a1 = -1.0 / (N * τn)
    a2 = -1.0 / τn

    dMxx = a1 * εxx + a2 * Mxxn
    dMzz = a1 * εzz + a2 * Mzzn
    dMxz = a1 * εxz + a2 * Mxzn
    return dMxx, dMzz, dMxz
end

@kernel function _stress_1Q!(s, M, v, λs, μs, τs, τns, D, dt)
    I = @index(Global, NTuple)
    pad = length(D)
    i, j = I[1] + pad, I[2] + pad
    N = length(τns)

    εxx = ddxm(i, j, 1, v, D)
    εzz = ddzm(i, j, 2, v, D)
    εxz = 0.5 * (_ddz(i, j, 1, v, D) + ddxp(i, j, 2, v, D))

    dsxx, dszz, dsxz = _ds_1Q(M, i, j, εxx, εzz, εxz, λs, μs, τs)
    s[i, j, 1] += dt * dsxx
    s[i, j, 2] += dt * dszz
    s[i, j, 3] += dt * dsxz
    
    for n in axes(M, 3)
        Mxx = M[i, j, n, 1]
        Mzz = M[i, j, n, 2]
        Mxz = M[i, j, n, 3]
        τn = τns[n]

        dMxx, dMzz, dMxz = _dM_1Q(Mxx, Mzz, Mxz, εxx, εzz, εxz, τn, N)
        M[i, j, n, 1] += dt * dMxx
        M[i, j, n, 2] += dt * dMzz
        M[i, j, n, 3] += dt * dMxz
    end
end

function leapfrog_step!(fields, parameters, dt)
    s = fields.s
    M = fields.M
    v = fields.v

    λs, μs, τs, τns = parameters

    device = get_backend(M)
    pad = length(D)
    worksize = (size(M, 1) - 2 * pad, size(M, 2) - 2 * pad)
    kernel = _stress_1Q!(device, worksize)
    kernel(s, M, v, λs, μs, τs, τns, D, dt)
    synchronize(device)
end

struct Fields
    s::Array
    M::Array
    v::Array
end

struct Parameters
    λs::Array
    μs::Array
    τs::Array
    τns::Array
end


