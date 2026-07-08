"""Differentiate the BADA3 atmosphere model with JAX."""

# --8<-- [start:input0]
import jax
import jax.numpy as jnp

from aerocore.bada3 import atmosphere
from aerocore.geo import G_0
from aerocore.thermo import R_SPECIFIC_DRY_AIR

altitude = jnp.linspace(0, 20_000, 100)
pressure_gradient = jax.vmap(
    jax.grad(
        lambda value: (
            atmosphere(
                value,
                delta_temperature=0.0,
                xp=jnp,
            ).pressure
        )
    )
)(altitude)
density = atmosphere(
    altitude,
    delta_temperature=0.0,
    xp=jnp,
).density(R_SPECIFIC_DRY_AIR)
print(jnp.allclose(pressure_gradient, -density * G_0))
# --8<-- [end:input0]

"""
--8<-- [start:output0]
True
--8<-- [end:output0]
"""
