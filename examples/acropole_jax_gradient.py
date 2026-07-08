# mypy: disable-error-code="arg-type,type-arg"
# --8<-- [start:input0]
from pathlib import Path

import isqx
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

import aerocore.types as t
from aerocore.acropole import fuel_flow, load_aircraft_database, load_model
from aerocore.utils import tree_map

MIN_GROUNDSPEED: t.GsKt = 250.0
MAX_GROUNDSPEED: t.GsKt = 500.0
MIN_ALTITUDE: t.PressureAltitudeFt = 20_000.0
MAX_ALTITUDE: t.PressureAltitudeFt = 40_000.0
kg_per_s_to_kg_per_h = isqx.convert(
    isqx.KG * isqx.S**-1, isqx.KG * isqx.HOUR**-1
)
GRID_SIZE = 401
ARROW_STRIDE = 16


device = jax.devices("gpu")[0]
model = tree_map(
    lambda array: jax.device_put(array, device=device), load_model()
)
aircraft = load_aircraft_database()["A320"]
mass = jax.device_put(65_000.0, device=device)
zero = jax.device_put(0.0, device=device)


def steady_cruise_fuel_flow(point: jax.Array) -> jax.Array:
    scaled_groundspeed, scaled_altitude = point
    groundspeed = MIN_GROUNDSPEED + scaled_groundspeed * (
        MAX_GROUNDSPEED - MIN_GROUNDSPEED
    )
    return fuel_flow(  # type: ignore[no-any-return]
        model=model,
        aircraft=aircraft,
        groundspeed=groundspeed,
        altitude=MIN_ALTITUDE + scaled_altitude * (MAX_ALTITUDE - MIN_ALTITUDE),
        vertical_rate=zero,
        airspeed=groundspeed,
        mass=mass,
        altitude_rate=zero,
        groundspeed_rate=zero,
        airspeed_rate=zero,
        xp=jnp,
    )


scaled_gs = jnp.linspace(0.0, 1.0, GRID_SIZE, device=device)
scaled_alt = jnp.linspace(0.0, 1.0, GRID_SIZE, device=device)
scaled_gs_grid, scaled_alt_grid = jnp.meshgrid(scaled_gs, scaled_alt)
points = jnp.stack((scaled_gs_grid, scaled_alt_grid), axis=-1)

ff_and_grad = jax.jit(
    jax.vmap(jax.value_and_grad(steady_cruise_fuel_flow)), device=device
)
ff, grad = ff_and_grad(points.reshape(-1, 2))
ff_grid = ff.reshape(GRID_SIZE, GRID_SIZE)
grad_grid = grad.reshape(GRID_SIZE, GRID_SIZE, 2)
unit_grad_grid = grad_grid / jnp.linalg.norm(grad_grid, axis=-1, keepdims=True)

gs = MIN_GROUNDSPEED + scaled_gs_grid * (MAX_GROUNDSPEED - MIN_GROUNDSPEED)
alt = MIN_ALTITUDE + scaled_alt_grid * (MAX_ALTITUDE - MIN_ALTITUDE)
unit_grad = unit_grad_grid * jnp.asarray(
    (MAX_GROUNDSPEED - MIN_GROUNDSPEED, MAX_ALTITUDE - MIN_ALTITUDE)
)

plt.style.use("dark_background")
figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
image = axis.imshow(
    kg_per_s_to_kg_per_h(ff_grid),
    origin="lower",
    extent=(MIN_GROUNDSPEED, MAX_GROUNDSPEED, MIN_ALTITUDE, MAX_ALTITUDE),
    aspect="auto",
    interpolation="bilinear",
)
contours = axis.contour(
    gs,
    alt,
    kg_per_s_to_kg_per_h(ff_grid),
    levels=10,
    colors="white",
    linewidths=0.8,
    alpha=0.7,
)
axis.clabel(contours, inline=True, fontsize=8, fmt="%.0f", colors="white")
axis.quiver(
    gs[::ARROW_STRIDE, ::ARROW_STRIDE],
    alt[::ARROW_STRIDE, ::ARROW_STRIDE],
    unit_grad[::ARROW_STRIDE, ::ARROW_STRIDE, 0],
    unit_grad[::ARROW_STRIDE, ::ARROW_STRIDE, 1],
    angles="xy",
    scale_units="xy",
    scale=32,
    width=0.0015,
)
axis.set_xlabel("Groundspeed (kt)")
axis.set_ylabel("Pressure altitude (ft)")
figure.colorbar(image, ax=axis, label="Fuel flow (kg/h)")
# --8<-- [end:input0]
output = Path("docs/assets/img/acropole-fuel-flow-gradient.png")
figure.savefig(output, dpi=180)
plt.close(figure)
