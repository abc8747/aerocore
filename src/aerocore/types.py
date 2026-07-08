"""This module contains a set of type aliases for `isqx` physical quantity kinds
and units.

## Usage

When annotating functions, use **unconstrained generic types**:

```py
from aerocore import types as t

def foo(x: t.StaticTemperatureK) -> t.PressurePA:
    ...
```

This signals to the static type checker that we want an `Unknown` type, not the
`Any` type. Do not annotate with `NDArray[np.float64]` or `float`.

Reasoning: we wish to support a wide range of numerical libraries, so
restricting ourselves to `numpy` will lead to many false positives in strict
mode. See https://docs.jax.dev/en/latest/jep/12049-type-annotations.html
for more information.
"""

from __future__ import annotations

from typing import Annotated, TypeVar

import isqx
import isqx.usc
from isqx import aerospace as aero

_T = TypeVar("_T")

#
# custom quantity kinds and metadata objects
#

TEMPERATURE_GRADIENT = isqx.QtyKind(
    isqx.K * isqx.M**-1, ("temperature_gradient",)
)
"""Temperature lapse rate, e.g., in an atmosphere model."""

BelowTropopause = "below_tropopause"
STATIC_TEMPERATURE_BELOW_TROPO = aero.STATIC_TEMPERATURE[BelowTropopause]
STATIC_PRESSURE_BELOW_TROPO = isqx.STATIC_PRESSURE[BelowTropopause]

DENSITY_FACTOR = isqx.Dimensionless("density_factor")
COMPRESSIBILITY_FACTOR = isqx.Dimensionless("compressibility_factor")
ENGINE = isqx.Dimensionless("engine")
ENGINE_TAKEOFF_FUEL_FLOW = isqx.QtyKind(
    isqx.KG * isqx.S**-1 * ENGINE**-1, ("takeoff",)
)

#
# Type Aliases
#

# base
LengthM = Annotated[_T, isqx.LENGTH(isqx.M)]
AngleRad = Annotated[_T, isqx.ANGLE(isqx.RAD)]
GravitationalAcceleration = Annotated[
    _T, isqx.ACCELERATION_OF_FREE_FALL(isqx.M_PERS2)
]
TimestampUtcS = Annotated[_T, isqx.TIME["timestamp", "utc"](isqx.S)]
DurationS = Annotated[_T, isqx.DURATION(isqx.S)]
MassKg = Annotated[_T, isqx.MASS(isqx.KG)]
LatitudeDeg = Annotated[_T, isqx.LATITUDE(isqx.DEG)]
LongitudeDeg = Annotated[_T, isqx.LONGITUDE(isqx.DEG)]

# thermodynamics
PressurePA = Annotated[_T, isqx.PRESSURE(isqx.PA)]
PressureHPA = Annotated[_T, isqx.PRESSURE(isqx.HECTO * isqx.PA)]
DensityKGM3 = Annotated[_T, isqx.DENSITY(isqx.KG * isqx.M**-3)]

StaticTemperatureK = Annotated[_T, aero.STATIC_TEMPERATURE(isqx.K)]
StaticPressurePA = Annotated[_T, isqx.STATIC_PRESSURE(isqx.PA)]
DynamicPressurePA = Annotated[_T, isqx.DYNAMIC_PRESSURE(isqx.PA)]
ImpactPressurePA = Annotated[_T, aero.IMPACT_PRESSURE(isqx.PA)]
TotalPressurePA = Annotated[_T, aero.TOTAL_PRESSURE(isqx.PA)]

GasConstantJMolK = Annotated[
    _T, isqx.MOLAR_GAS_CONSTANT(isqx.J * isqx.MOL**-1 * isqx.K**-1)
]
MolarMassKGMol = Annotated[_T, isqx.MOLAR_MASS(isqx.KG * isqx.MOL**-1)]
SpecificGasConstantJKGK = Annotated[
    _T, isqx.SPECIFIC_GAS_CONSTANT(isqx.J * isqx.KG**-1 * isqx.K**-1)
]
TemperatureGradientKPM = Annotated[
    _T, TEMPERATURE_GRADIENT(isqx.K * isqx.M**-1)
]
StaticTemperatureKBelowTropo = Annotated[
    _T, STATIC_TEMPERATURE_BELOW_TROPO(isqx.K)
]
StaticPressurePABelowTropo = Annotated[_T, STATIC_PRESSURE_BELOW_TROPO(isqx.PA)]

# aerospace
CasMPS = Annotated[_T, aero.CALIBRATED_AIRSPEED(isqx.M_PERS)]
EasMPS = Annotated[_T, aero.EQUIVALENT_AIRSPEED(isqx.M_PERS)]
TasMPS = Annotated[_T, aero.TRUE_AIRSPEED(isqx.M_PERS)]
TasKt = Annotated[_T, aero.TRUE_AIRSPEED(isqx.usc.KNOT)]
MaximumOperatingTasKt = Annotated[
    _T, aero.TRUE_AIRSPEED["maximum_operating"](isqx.usc.KNOT)
]
GsKt = Annotated[_T, aero.GROUND_SPEED(isqx.usc.KNOT)]
SpeedOfSoundMPS = Annotated[_T, isqx.SPEED_OF_SOUND(isqx.M_PERS)]

GeopotentialAltitudeM = Annotated[_T, aero.GEOPOTENTIAL_ALTITUDE(isqx.M)]
PressureAltitudeFt = Annotated[_T, aero.PRESSURE_ALTITUDE(isqx.usc.FT)]
MaximumOperatingPressureAltitudeFt = Annotated[
    _T, aero.PRESSURE_ALTITUDE["maximum_operating"](isqx.usc.FT)
]
VerticalRateFtMin = Annotated[
    _T, aero.VERTICAL_RATE(isqx.usc.FT * isqx.MIN**-1)
]
VerticalRateFtS = Annotated[_T, aero.VERTICAL_RATE(isqx.usc.FT * isqx.S**-1)]
AccelerationKtS = Annotated[_T, isqx.ACCELERATION(isqx.usc.KNOT * isqx.S**-1)]
GeometricAltitudeM = Annotated[_T, aero.GEOMETRIC_ALTITUDE(isqx.M)]
WingAreaM2 = Annotated[
    _T, aero.WING_AREA(isqx.M**2)
]  # TODO: Wimpress or airbus?
OperatingEmptyWeightKg = Annotated[_T, aero.OPERATING_EMPTY_WEIGHT(isqx.KG)]
MaximumTakeoffWeightKg = Annotated[_T, aero.MAXIMUM_TAKEOFF_WEIGHT(isqx.KG)]
ThrustN = Annotated[_T, isqx.FORCE(isqx.N)]
MassFlowKgPS = Annotated[_T, isqx.MASS_FLOW_RATE(isqx.KG * isqx.S**-1)]
MassFlowKgPSPEngine = Annotated[
    _T,
    ENGINE_TAKEOFF_FUEL_FLOW(isqx.KG * isqx.S**-1 * ENGINE**-1),
]
SpecificFuelConsumptionKgPNPS = Annotated[
    _T,
    aero.THRUST_SPECIFIC_FUEL_CONSUMPTION(isqx.KG * isqx.N**-1 * isqx.S**-1),
]

# dimensionless
MachNumber = Annotated[_T, isqx.MACH_NUMBER]
RatioOfSpecificHeats = Annotated[_T, isqx.HEAT_CAPACITY_RATIO]
DensityFactor = Annotated[_T, DENSITY_FACTOR]
CompressibilityFactor = Annotated[_T, COMPRESSIBILITY_FACTOR]
BypassRatio = Annotated[_T, aero.BYPASS_RATIO]

# deltas
DeltaTemperatureK = Annotated[_T, isqx.TEMPERATURE[isqx.DELTA](isqx.K)]
DeltaLengthM = Annotated[_T, isqx.LENGTH[isqx.DELTA](isqx.M)]

# datalink: acars / vhf / vdl2 / hfdl
FrequencyHz = Annotated[_T, isqx.FREQUENCY(isqx.HZ)]
FrequencyMhz = Annotated[_T, isqx.FREQUENCY(isqx.MEGA * isqx.HZ)]
BitPS = Annotated[_T, isqx.BIT_RATE(isqx.BIT * isqx.S**-1)]
