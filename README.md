# Ghostlight

Can light from a nearby star lead to a misleading measurement of gravitational lensing?

## What astronomers observe

A black hole can be difficult to detect when it has no bright gas around it. Its gravity can still bend light from a star behind it. As the objects move into alignment, the background star can appear brighter and its apparent position can shift. This is called gravitational microlensing.

The telescope receives light from the background star. The black hole is detected through its effect on that light. This project focuses on the small change in the star's apparent position.

## Why that measurement matters

The amount of bending carries information about the unseen object. Combined with additional distance information, it can help estimate the object's mass and assess whether it could be a black hole. A positional shift alone does not establish that.

A nearby star can complicate the measurement. If a telescope cannot distinguish the two stars, their combined light affects the position it records. Treating that combined light incorrectly can produce a misleading estimate of the lensing effect.

## What this project actually does

This version simulates the background star, the lensing effect and light from a second star. It compares a model that allows the second star to occupy its own position with one that forces it onto the background star's position.

The incorrect model misses the known lensing value more often while reporting almost the same uncertainty. It also leaves a clear pattern in the differences between its predictions and the simulated positions. Checking those differences should expose the incorrect model.

The result shows why small reported errors are insufficient evidence that a lensing measurement is trustworthy. It uses simulated measurements; it has not found a black hole or measured the mass of a real object.

![Calculated differences between each model and the known simulated star positions; these curves are not telescope observations](figures/ghostlight.png)

[Read the case study](https://alisonthomasportfolio.com/projects/ghostlight) · [Explore the code](https://github.com/alithomas321/ghostlight/tree/main/study/src/ghostlight) · [Methods](study/METHODOLOGY.md)

<details>
<summary>Run the code</summary>

With Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
cd study
uv sync --frozen --extra dev
uv run pytest
uv run ghostlight verify
```

The saved figures and tables are in study/outputs. See [reproduction details](study/REPRODUCIBILITY.md) for the full simulation.

</details>

Background: [NASA: measuring an isolated black hole through its effect on starlight](https://science.nasa.gov/missions/hubble/hubble-determines-mass-of-isolated-black-hole-roaming-our-milky-way-galaxy/). This describes the astronomy; Ghostlight does not analyze that observed event.
