import ViscoelasticInversion: find_optimal_τns, K, get_τ, get_Q

function test_const_Q_optim(ω1=2π*3, ω2=2π*15, Q0 = 100, N = 5)
    # Tests const Q model optimization with the same params as Fichtner 2014
    ωs = range(ω1, ω2, length=100)
    Δω = (ω2-ω1)/length(ωs)
    τns = find_optimal_τns(ω1, ω2, N)
    Ks = K.(ωs, Ref(τns))
    τ = get_τ(Q0, mean(Ks), N)
    Qs = get_Q.(ωs, Ref(τ), Ref(τns))

    println("ω1: $ω1")
    println("ω2: $ω2")
    println("Q0: $Q0")
    println("N: $N")
    println("Optimal τns: $τns")
    println("Mean relative error: $(round(mean(abs.(Qs .- Q0))/Q0*100, digits=2)) %")
    println("Maximum relative error: $(round(maximum(abs.(Qs .- Q0)) / Q0 * 100, digits=2)) %")
end

# test_const_Q_optim(2π*3, 2π*15, 100, 5)
# test_const_Q_optim(2π*0.1, 2π*1.0, 100, 5)