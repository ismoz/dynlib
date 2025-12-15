# An interesting animation that I like.

import numpy as np
from dynlib.plot import export, vectorfield_animate


DSL = """
inline:
[model]
type="ode"

[states]
x=0.0
y=0.0

[params]
k=0.0

[equations.rhs]
x = "sin(k*y)"
y = "sin(k*x)"
"""

# You have to define anim (or any other name) even if you don't use it. 
# Otherwise it gets garbage collected.
anim=vectorfield_animate(DSL, 
                    param="k", 
                    values=np.linspace(0.1,10,300), 
                    xlim=(-10, 10), 
                    ylim=(-10, 10), 
                    grid=(24, 24), 
                    interval=130,
                    normalize=True,
                    title_func=lambda v, idx: f"Vector field: k={float(v):.2f}",
                    )

# Save using writer of your choice, e.g., "ffmpeg", "pillow", etc.
# anim.save("vectorfield_animation.gif", writer="pillow", dpi=150)

export.show()
