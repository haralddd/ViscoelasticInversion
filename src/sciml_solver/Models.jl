using KernelAbstractions
include(joinpath("..", "utils.jl"))

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

