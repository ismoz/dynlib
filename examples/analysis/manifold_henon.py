from dynlib import setup
from dynlib.analysis import trace_manifold_1d_map
from dynlib.plot import manifold, theme, export, fig


a, b = 1.4, 0.3
bounds = ((-2.5, 2.5), (-2.5, 2.5))

sim = setup("builtin://map/henon2", jit=True)
sim.assign(a=1.4, b=0.3)

fp = sim.model.fixed_points(seeds=[[0.8, 0.8]])
stable_results = trace_manifold_1d_map( sim, 
                                        fp=fp.points[0],
                                        bounds=bounds,
                                        kind="stable",
                                        steps=80,
                                        hmax=2e-3,
                                        clip_margin=0.35,
                                        max_points_per_segment=40000,
                                        max_segments=300,
                                    )

unstable_results = trace_manifold_1d_map( sim, 
                                          fp=fp.points[0],
                                          bounds=bounds,
                                          kind="unstable",
                                          steps=80,
                                          hmax=2e-3,
                                          clip_margin=0.35,
                                          max_points_per_segment=120000,
                                          max_segments=400,
                                        )

theme.use("paper")
ax = fig.single(size=(5, 5))
manifold(
    result=stable_results,
    ax=ax,
    color="blue",
    label="stable",
    xlim=bounds[0],
    ylim=bounds[1],
    xlabel="$x$",
    ylabel="$y$",
    title="Henon map manifolds",
    aspect="equal",
    lw=0.7,
)
manifold(result=unstable_results, ax=ax, color="red", label="unstable", lw=0.7)
ax.plot(fp.points[0][0], fp.points[0][1], "k+", ms=12)
export.show()
