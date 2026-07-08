"""Evaluate aerocore expressions lazily with Polars."""

# --8<-- [start:input0]
import polars as pl

import numpy as np
from aerocore.bada3 import atmosphere
from aerocore.polars import PolarsArrayApiNamespace


def calc_atmosphere(altitude: pl.Expr) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    result = atmosphere(
        altitude, delta_temperature=0.0, xp=PolarsArrayApiNamespace
    )
    return (
        altitude,
        result.pressure.alias("pressure"),
        result.temperature.alias("temperature"),
    )


frame = pl.DataFrame({"altitude": np.linspace(0, 20_000, 5)}).lazy()
query = frame.select(calc_atmosphere(pl.col("altitude")))
print(query.explain(format="tree"))
# --8<-- [end:input0]

"""
--8<-- [start:output0]
               0                                                         1                                                                     2                                                    3                                                      4                                     5
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   │
   │       ╭────────╮
 0 │       │ SELECT │
   │       ╰───┬┬───╯
   │           ││
   │           │╰────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────╮
   │           │                                                         │                                                                     │                                                    │
   │           │           ╭─────────────────────────────────────────────┴─────────────────────────────────────────────╮                       │                                                    │
   │           │           │ expression:                                                                               │                       │                                                    │
   │           │           │ when(col("__POLARS_CSER_0x7dbd3b72cd594ba5"))                                             │                       │                                                    │
   │  ╭────────┴────────╮  │   .then([(101325.0) * ([([(col("__POLARS_CSER_0x48a7fecdc84dc074")) - (0.0)]) / (288.15)] │  ╭────────────────────┴────────────────────╮                        ╭──────┴───────╮
   │  │ expression:     │  │   .alias("literal")                                                                       │  │ expression:                             │                        │ FROM:        │
 1 │  │ col("altitude") │  │   .pow([dyn float: 5.25587973947749]))])                                                  │  │ col("__POLARS_CSER_0x48a7fecdc84dc074") │                        │ WITH_COLUMNS │
   │  ╰─────────────────╯  │   .otherwise([(22632.06) * ([([(col("altitude")) - (11000.0)]) * (-0.000158)]             │  │   .alias("temperature")                 │                        ╰──────┬┬──────╯
   │                       │   .exp())])                                                                               │  ╰─────────────────────────────────────────╯                               ││
   │                       │   .alias("pressure")                                                                      │                                                                            ││
   │                       ╰───────────────────────────────────────────────────────────────────────────────────────────╯                                                                            ││
   │                                                                                                                                                                                                ││
   │                                                                                                                                                                                                │╰─────────────────────────────────────────────────────┬─────────────────────────────────────╮
   │                                                                                                                                                                                                │                                                      │                                     │
   │                                                                                                                                                                   ╭────────────────────────────┴────────────────────────────╮                         │                                     │
   │                                                                                                                                                                   │ expression:                                             │  ╭──────────────────────┴───────────────────────╮  ╭──────────┴──────────╮
   │                                                                                                                                                                   │ when([(col("altitude")) <= (11000.0)])                  │  │ expression:                                  │  │ DF ["altitude"]     │
 2 │                                                                                                                                                                   │   .then([(288.15) + ([(col("altitude")) * (-0.0065)])]) │  │ [(col("altitude")) <= (11000.0)]             │  │ PROJECT */1 COLUMNS │
   │                                                                                                                                                                   │   .otherwise(216.65)                                    │  │   .alias("__POLARS_CSER_0x7dbd3b72cd594ba5") │  ╰─────────────────────╯
   │                                                                                                                                                                   │   .alias("__POLARS_CSER_0x48a7fecdc84dc074")            │  ╰──────────────────────────────────────────────╯
   │                                                                                                                                                                   ╰─────────────────────────────────────────────────────────╯
--8<-- [end:output0]
"""

# --8<-- [start:input1]
print(query.collect())
# --8<-- [end:input1]

"""
--8<-- [start:output1]
shape: (5, 3)
┌──────────┬──────────────┬─────────────┐
│ altitude ┆ pressure     ┆ temperature │
│ ---      ┆ ---          ┆ ---         │
│ f64      ┆ f64          ┆ f64         │
╞══════════╪══════════════╪═════════════╡
│ 0.0      ┆ 101325.0     ┆ 288.15      │
│ 5000.0   ┆ 54019.888662 ┆ 255.65      │
│ 10000.0  ┆ 26436.243088 ┆ 223.15      │
│ 15000.0  ┆ 12044.563506 ┆ 216.65      │
│ 20000.0  ┆ 5474.882348  ┆ 216.65      │
└──────────┴──────────────┴─────────────┘
--8<-- [end:output1]
"""
