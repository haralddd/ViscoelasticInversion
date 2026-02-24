abstract type AbstractModel end

@kwdef struct IsotropicModel <: AbstractModel
    b
    C11
    C13
    C33
    C55

    "Constructs homogeneous isotropic model from Lamé parameters"
    function IsotropicModel(ρ::T, λ::T, μ::T, Nx, Nz; device=CPU()) where T <: Real
        F = preferred_float(device)
        b = KA.zeros(device, F, Nx,Nz)
        C11 = similar(b)
        C33 = similar(b)
        C13 = similar(b)
        C55 = similar(b)

        fill!(b, 1/ρ)
        fill!(C11, λ+2μ)
        fill!(C33, λ+2μ)
        fill!(C13, λ)
        fill!(C55, μ)

        return new(b, C11, C13, C33, C55)
    end
end

@kwdef struct VTIModel <: AbstractModel
    b
    C11
    C13
    C33
    C55
end

@kwdef struct TTIModel <: AbstractModel
    b
    C11
    C13
    C15
    C33
    C35
    C55
end

size(model::T) where T <: AbstractModel = size(model.b)


abstract type AbstractViscoModel end
# Optimization of visco-coefficients
K(ω, τns) = sum(ω*τn / (1.0 + τn^2 * ω^2) for τn in τns)
dK_dω(ω, τns) = sum( τn * (1.0 - τn^2 * ω^2) / (1.0 + τn^2 * ω^2)^2 for τn in τns)

function Q_model_J2(τns, ω1, ω2)
    integrand(ω) = dK_dω(ω, τns)^2
    J2, _ = quadgk(integrand, ω1, ω2)
    return J2
end

"Finds optimal τns for const Q model and returns vector of them"
function find_optimal_τns(ω1, ω2, N)

    # Log spaced initial guess
    τns_init = 1.0 ./ range(ω1, ω2, length=N)
    result = optimize(
        τns -> Q_model_J2(τns, ω1, ω2),
        τns_init,
        NelderMead(),
        Optim.Options(iterations=10^6)
    )
    return sort(result.minimizer)
end

get_τ(Q0,K,N) = N / (K*Q0)
get_Q(ω, τ, τns) = length(τns) / (τ * K(ω,τns))

function Q_mean_re(Qs, Q0s)
    return round(mean(abs.(Qs .- Q0s) / Q0s), digits=2) * 100
end

function Q_max_re(Qs, Q0s)
    return round(maximum(abs.(Qs .- Q0s) / Q0s), digits=2) * 100
end

# Nearly constant Q models using the method described by Fichtner 2014
"""
Nearly constant Q model using the method described by Fichtner (2014 pg. 87).


"""
@kwdef struct OneQModel{T} <: AbstractViscoModel
    basemodel::T
    τ
    τns

    # First time init: Optimizes coeffs τn based on frequency range f1..f2
    function OneQModel(basemodel::T, f1, f2, N=5, Q0s=nothing) where T
        if Q0s === nothing
            Q0s = fill(100, size(basemodel))
        end
        @assert size(Q0s) == size(basemodel)

        τns = find_optimal_τns(2π*f1, 2π*f2, N)
        Ks = K.(range(2π*f1, 2π*f2, length=100), Ref(τns))
        K = mean(Ks)
        τ = get_τ.(Q0s, K, N)

        return new{T}(basemodel, τ, τns, f1, f2)
    end

    # Subsequent inits: Uses provided τns for same frequency range, with new Q
    function OneQModel(basemodel::T, τ, τns) where T
        return new{T}(basemodel, τ, τns)
    end
end

@kwdef struct TwoQModel{T} <: AbstractViscoModel
    basemodel::T
    τp
    τs
    τpn
    τsn

    function TwoQModel(basemodel::T, fp1, fp2, fs1, fs2, Np=5, Ns=5, Qp0=100, Qp0s=nothing, Qs0=100, Qs0s=nothing) where T
        if Qp0s === nothing
            Qp0s = fill(Qp0, size(basemodel))
        end
        if Qs0s === nothing
            Qs0s = fill(Qs0, size(basemodel))
        end
        @assert size(Qp0s) == size(basemodel)
        @assert size(Qs0s) == size(basemodel)

        τpn = find_optimal_τns(2π*fp1, 2π*fp2, Np)
        Ks = K.(range(2π*fp1, 2π*fp2, length=100), Ref(τpn))
        K = mean(Ks)
        τp = get_τ.(Qp0s, K, Np)

        τsn = find_optimal_τns(2π*fs1, 2π*fs2, Ns)
        Ks = K.(range(2π*fs1, 2π*fs2, length=100), Ref(τsn))
        K = mean(Ks)
        τs = get_τ.(Qs0s, K, Ns)
        return new{T}(basemodel, τp, τs, τpn, τsn)
    end
end



