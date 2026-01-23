using SegyIO
using CairoMakie

model_p = segy_read("elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy")

data = Float32.(model_p.data)
fig = Figure(resolution=(1191, 1684))
ax = fig[1, 1] = Axis(fig)
heatmap!(ax, rotr90(data), colormap=:magma)
hidedecorations!(ax)
display(fig)
save("model_p.png", fig)