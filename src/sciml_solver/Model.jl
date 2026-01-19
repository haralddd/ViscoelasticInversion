using KernelAbstractions

abstract type AbstractModel end

@kwdef struct IsotropicModel <: AbstractModel
    b
    C11
    C13
    C33
    C55

    "Constructs homogeneous isotropic model from Lamé parameters"
    function IsotropicModel(ρ::T, λ::T, μ::T, Nx, Nz) where T <: Real
        b = fill(1/ρ, Nx, Nz)
        C11 = C33 = fill(λ + 2μ, Nx, Nz)
        C13 = fill(λ, Nx, Nz)
        C55 = fill(μ, Nx, Nz)
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

