module ViscoelasticInversion

using Roots
using OrdinaryDiffEq

import KernelAbstractions as KA
import Base./ # For joinpath(string...) convenience

include("common/utils.jl")
include("common/Stencil.jl")

include("sciml_solver/Preallocated.jl")
include("sciml_solver/Source.jl")
include("sciml_solver/BoundaryCondition.jl")
include("sciml_solver/Model.jl")
include("sciml_solver/NeuralPDEModel.jl")
include("sciml_solver/ViscoelasticProblem.jl")

export Preallocated
export RickerSource
export AbstractBC, DirichletBC, NeumannBC, PeriodicBC, AbsorbingBC
export ViscoelasticProblem

end # module