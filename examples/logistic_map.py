from dynlib import setup
from dynlib.plot import series, export, theme, fig, return_map, cobweb

model = '''
inline:
[model]
type="map"
name="Logistic Map"

[states]
x=0.1

[params]
r=4.0

[equations.rhs]
x = "r * x * (1 - x)"
'''

# stepper="map" is default and can be omitted for map models
sim = setup(model, stepper="map", jit=True, disk_cache=True)
sim.run(N=192, transient=40)
sim.run(resume=True, N=400)
res=sim.results()

theme.use("notebook")
theme.update(grid=False)

# Create 1x2 grid for time series and return map
ax = fig.grid(rows=1, cols=2, size=(12, 5))

# Time series plot
series.plot(
    x=res.t,
    y=res["x"],
    style="line",
    ax=ax[0, 0],
    xlabel="n",
    ylabel="$x_n$",
    ylabel_rot=0,
    title="Logistic Map (r=4)",
    ypad=10,
    xlabel_fs=13,
    ylabel_fs=13,
    title_fs=14,
    xtick_fs=9,
    ytick_fs=11,
    lw=1.0
)

# Return map: x[n] vs x[n+1]
return_map(
    x=res["x"],
    step=1,
    style="scatter",
    ax=ax[0, 1],
    ms=2,
    color="C1",
    title="Return Map: $x[n]$ vs $x[n+1]$",
    xlabel_fs=13,
    ylabel_fs=13,
    title_fs=14,
    xtick_fs=9,
    ytick_fs=11,
)

cobweb(
    f=sim.model,
    x0=0.1,
    xlim=(0, 1),
    steps=50,
    color="green",
    stair_color="orange",
    identity_color="red",
)

export.show()
