---
name: generation-aligned-charts
description: Every training-history chart (accuracy, loss, parameters or anything else per epoch) must show where each generation ends, with the x axis scaled to the generation length. Use when drawing or changing any per-epoch chart of a growingnn run, in a report generator in documentation/website/scripts/, an experiment script or a quick chart for the user.
---

# Generation-aligned training charts

A growingnn run trains in generations and changes the architecture between them. A training chart is
only readable if the reader can see where each generation ends, because that is where an action was
applied and where jumps, dips or slope changes come from.

## Rules

1. **Read the generation length from the run**, never assume it: `epochs_per_generation` in the run's
   `manifest.json`, or `epochInGeneration` / `generation` in `board/metrics/training.json`. Runs differ
   (Experiment 013 Phase 17 has 20 epochs per generation, Phase 18 has 30).
2. **The x axis is scaled to generations.** Put the major ticks and the vertical grid lines at the
   generation boundaries (multiples of the generation length): 0, 30, 60, ... for 30 epochs per
   generation. Do not leave matplotlib's default ticks (0, 50, 100, ...), which do not match the
   generations and hide them.
3. **The vertical gray lines are the generation boundaries and nothing else.** Draw one at the end of
   every generation, including when the run is still going (draw them for the whole planned run, set
   `xlim` to the planned number of epochs). Turn off the default vertical grid so no other gray line
   competes with them; a horizontal grid for the y values is fine.
4. **Label the generations.** Add a secondary x axis on top (`ax.secondary_xaxis('top', ...)`) with
   the generation numbers centred in each generation, or name them in the caption when there is no
   room. With many generations (more than about 15), label every second or fifth one but keep a line
   at every boundary.
5. **Several runs with different generation lengths in one chart:** use small multiples (one panel per
   generation length) instead of mixing two sets of boundaries in one panel.
6. **Shared x axis panels** (accuracy above, parameters below) get the same boundaries in every panel.

## Matplotlib recipe

```python
from matplotlib.ticker import MultipleLocator

def mark_generations(axes, epochs_per_generation, generations):
    """Ticks, gray lines and generation labels at every generation boundary."""
    total = epochs_per_generation * generations
    for axis in axes:
        axis.set_xlim(0, total)
        axis.xaxis.set_major_locator(MultipleLocator(epochs_per_generation))
        axis.grid(False, axis='x')
        axis.grid(True, axis='y', alpha=.25)
        for boundary in range(epochs_per_generation, total, epochs_per_generation):
            axis.axvline(boundary + .5, color='#cccccc', lw=.8, zorder=0)
    top = axes[0].secondary_xaxis('top')
    top.set_xticks([epochs_per_generation * (g + .5) for g in range(generations)],
                   [str(g) for g in range(generations)])
    top.tick_params(length=0, labelsize=7)
    top.set_xlabel('Generation')
```

The `+ .5` puts the line between the last epoch of a generation and the first epoch of the next one,
when epochs are plotted as `globalEpoch + 1`.

## Check before sending

Look at the rendered chart: the gray lines must fall exactly where the curves jump after an action,
the ticks must be multiples of the generation length, and the caption must say how many epochs a
generation has.
