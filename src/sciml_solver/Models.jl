abstract type AbstractModel end

@kwdef struct IsotropicModel <: AbstractModel
    b
    λ
    μ

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


abstract type AbstractQModel end
struct ZeroQModel <: AbstractQModel end
# Nearly constant Q models using the method described by Fichtner 2014
"""
Nearly constant Q model using the method described by Fichtner (2014 pg. 87).


"""
@kwdef struct QModel <: AbstractQModel
    τ
    τns
    f1
    f2
    K
end

# Optimization of visco-coefficients
K(ω, τns) = sum(ω*τn / (1.0 + τn^2 * ω^2) for τn in τns)
dK_dω(ω, τns) = sum( τn * (1.0 - τn^2 * ω^2) / (1.0 + τn^2 * ω^2)^2 for τn in τns)
get_τ(Q0, K, N) = N / (K*Q0)
get_Q(ω, τ, τns) = length(τns) / (τ * K(ω,τns))

function Q_model_J2(τns, ω1, ω2)
    integrand(ω) = dK_dω(ω, τns)^2
    J2, _ = quadgk(integrand, ω1, ω2)
    return J2
end

"Finds optimal τns for const Q model and returns vector of them"
function find_optimal_τns(ω1, ω2, N)
    # Initial guess: 1/f spacing
    τns_init = 1.0 ./ range(ω1, ω2, length=N)
    result = optimize(
        τns -> Q_model_J2(τns, ω1, ω2),
        τns_init,
        NelderMead(),
        Optim.Options(iterations=10^6)
    )
    return sort(result.minimizer)
end

# First time init: Optimizes coeffs τn based on frequency range f1..f2
function QModel(f1, f2, Nx, Nz, Q0=100, N=5, device=CPU())
    _τns = find_optimal_τns(2π*f1, 2π*f2, N)
    Ks = K.(range(2π*f1, 2π*f2, length=100), Ref(_τns))
    K = mean(Ks)
    _τ = get_τ(Q0, K, N)

    τ = KA.zeros(device, preferred_float(device), Nx, Nz)
    fill!(τ, _τ)
    τns = KA.zeros(device, preferred_float(device), N)
    copyto!(τns, _τns)

    return QModel(τ, τns, f1, f2, K)
end

function update!(model::QModel, Q::Matrix)
    model.τ .= get_τ.(Q, model.K, length(model.τns))
    return nothing
end



