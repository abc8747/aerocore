# --8<-- [start:input0]
import jax
import jax.numpy as jnp

import numpy as np
from aerocore.acropole import fuel_flow, load_aircraft_database, load_model
from aerocore.utils import tree_map

device = jax.devices("gpu")[0]
model = tree_map(lambda array: jnp.asarray(array, device=device), load_model())
aircraft = load_aircraft_database()["A320"]


def predict(
    groundspeed: jax.Array, altitude: jax.Array, vertical_rate: jax.Array
) -> jax.Array:
    return fuel_flow(  # type: ignore[no-any-return]
        model=model,
        aircraft=aircraft,
        groundspeed=groundspeed,
        altitude=altitude,
        vertical_rate=vertical_rate,
        xp=jnp,
    )


predict = jax.jit(predict, device=device)
print(
    np.asarray(
        predict(
            groundspeed=jnp.asarray([180.0, 450.0, 250.0], device=device),
            altitude=jnp.asarray([0.0, 30_000.0, 40_000.0], device=device),
            vertical_rate=jnp.asarray([3_000.0, 0.0, -2_000.0], device=device),
        )
    )
)
# --8<-- [end:input0]

"""
--8<-- [start:output0]
[1.9125862  0.75758696 0.09448687]
--8<-- [end:output0]
"""
