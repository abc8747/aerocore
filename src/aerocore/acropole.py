# mypy: disable-error-code="no-untyped-def"
"""Array API implementation of the Acropole fuel-flow model."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from . import types as t
from .array_api import ArrayApiNamespace
from .utils import default_cache_dir

ACROPOLE_DATA_VERSION = "0.1.0"
ACROPOLE_MODEL_FILENAME = "acropole_model.npz"
ACROPOLE_AIRCRAFT_FILENAME = "acropole_aircraft.csv"
ACROPOLE_LICENSE_FILENAME = "acropole_LICENSE.txt"
ACROPOLE_RELEASE_BASE_URL = (
    "https://github.com/abc8747/aerocore/releases/download/v0.2.1"
)


def data_dir(cache_dir: Path | None = None) -> Path:
    return (
        (cache_dir or default_cache_dir())
        / "acropole"
        / f"v{ACROPOLE_DATA_VERSION}"
    )


async def sync_data(
    cache_dir: Path | None = None,
    force: bool = False,
    base_url: str = ACROPOLE_RELEASE_BASE_URL,
) -> tuple[Path, ...]:
    """Download the prepared Acropole model, aircraft, and license files.

    Requires optional dependency `httpx`.
    """
    root = data_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = tuple(
        root / filename
        for filename in (
            ACROPOLE_MODEL_FILENAME,
            ACROPOLE_AIRCRAFT_FILENAME,
            ACROPOLE_LICENSE_FILENAME,
        )
    )
    pending = (
        paths if force else tuple(path for path in paths if not path.exists())
    )
    if not pending:
        return paths

    import httpx

    async with httpx.AsyncClient(follow_redirects=True) as client:
        for destination in pending:
            response = await client.get(
                f"{base_url.rstrip('/')}/{destination.name}"
            )
            response.raise_for_status()
            temporary = destination.with_name(f".{destination.name}.part")
            temporary.write_bytes(response.content)
            temporary.replace(destination)
    return paths


#
# model
#


ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class AcropoleStandardisation(Generic[ArrayT]):
    minimums: ArrayT
    maximums: ArrayT


@dataclass(frozen=True)
class AcropoleWeights(Generic[ArrayT]):
    weights: tuple[ArrayT, ...]
    biases: tuple[ArrayT, ...]


@dataclass(frozen=True)
class AcropoleModel(Generic[ArrayT]):
    standardisation: AcropoleStandardisation[ArrayT]
    weights: AcropoleWeights[ArrayT]


def load_model(
    path: Path | None = None,
) -> AcropoleModel[npt.NDArray[np.float32]]:
    """Load model arrays from the Acropole cache.

    Use [`aerocore.utils.tree_map`][] to move to another device or type.
    """
    model_path = path or data_dir() / ACROPOLE_MODEL_FILENAME
    with np.load(model_path, allow_pickle=False) as archive:
        return AcropoleModel(
            standardisation=AcropoleStandardisation(
                minimums=archive["feature_minimums"],
                maximums=archive["feature_maximums"],
            ),
            weights=AcropoleWeights(
                weights=tuple(archive[f"weight_{index}"] for index in range(5)),
                biases=tuple(archive[f"bias_{index}"] for index in range(5)),
            ),
        )


#
# aircraft
#


class AcropoleEngineType(IntEnum):
    JET = 0
    TURBOPROP = 1


@dataclass(frozen=True)
class AcropoleAircraft:
    engine_type: AcropoleEngineType
    wing_area: t.WingAreaM2[float]
    max_alt: t.MaximumOperatingPressureAltitudeFt[float]
    max_tas: t.MaximumOperatingTasKt[float]
    oew: t.OperatingEmptyWeightKg[float]
    max_tow: t.MaximumTakeoffWeightKg[float]
    fuel_flow_per_engine_to: t.MassFlowKgPSPEngine[float]
    engine_count: int
    # following are unused in inference but keeping it for reference
    representative_engine: str | None = None
    trained: bool | None = None


IcaoTypeCode: TypeAlias = str
AcropoleAircraftDatabase: TypeAlias = dict[IcaoTypeCode, AcropoleAircraft]


def _decode_aircraft_database(
    lines: Iterable[str],
) -> AcropoleAircraftDatabase:
    return {
        row["ACFT_ICAO_TYPE"]: AcropoleAircraft(
            engine_type=AcropoleEngineType(int(row["ENGINE_TYPE"])),
            wing_area=float(row["SURFACE"]),
            max_alt=float(row["MAX_OPE_ALTI"]),
            max_tas=float(row["MAX_OPE_SPEED"]),
            oew=float(row["OPE_EMPTY_WEIGHT"]),
            max_tow=float(row["MAX_TO_WEIGHT"]),
            fuel_flow_per_engine_to=float(row["FUEL_FLOW_TO"]),
            engine_count=int(row["ENGINE_NUM"]),
            representative_engine=row["ENGINE_ICAO"],
            trained=bool(int(row["TRAIN_ON"])),
        )
        for row in csv.DictReader(lines)
    }


def load_aircraft_database(
    source: Path | Iterable[str] | None = None,
) -> AcropoleAircraftDatabase:
    if source is None:
        source = data_dir() / ACROPOLE_AIRCRAFT_FILENAME
    if isinstance(source, Path):
        with source.open(newline="", encoding="utf-8") as file:
            return _decode_aircraft_database(file)
    return _decode_aircraft_database(source)


#
# predict
#


def fuel_flow(
    model: AcropoleModel,
    aircraft: AcropoleAircraft,
    groundspeed: t.GsKt,
    altitude: t.PressureAltitudeFt,
    vertical_rate: t.VerticalRateFtMin,
    *,
    airspeed: t.TasKt | None = None,
    mass: t.MassKg | None = None,
    elapsed_time: t.DurationS | None = None,
    altitude_rate: t.VerticalRateFtS | None = None,
    groundspeed_rate: t.AccelerationKtS | None = None,
    airspeed_rate: t.AccelerationKtS | None = None,
    xp: ArrayApiNamespace,
) -> t.MassFlowKgPS:
    """Predict fuel flow from caller-owned backend arrays.

    Trajectory arrays must have identical shapes. The final axis represents
    samples ordered in time. When `elapsed_time` is provided, it must have the
    same shape, contain at least two samples, and be strictly increasing without
    duplicate timestamps.

    The paper reports one-second QAR samples but does not specify a
    resampling interval or smoothing-window duration. The upstream
    implementation later recommended approximately four-second sampling.
    Callers are responsible for resampling, interpolation, and
    smoothing before calling this function.
    """
    features = input_features(
        aircraft,
        groundspeed,
        altitude,
        vertical_rate,
        airspeed=airspeed,
        mass=mass,
        elapsed_time=elapsed_time,
        altitude_rate=altitude_rate,
        groundspeed_rate=groundspeed_rate,
        airspeed_rate=airspeed_rate,
        xp=xp,
    )
    standardised = standardise(model.standardisation, features)
    normalised_flow = predict_standardised(model.weights, standardised, xp=xp)
    fuel_scale = aircraft.fuel_flow_per_engine_to * aircraft.engine_count
    return normalised_flow * fuel_scale


# low level functions


def _diff_bfill(values, *, xp: ArrayApiNamespace):
    differences = xp.diff(values, axis=-1)
    return xp.concat((differences[..., :1], differences), axis=-1)


def input_features(
    aircraft: AcropoleAircraft,
    groundspeed: t.GsKt,
    altitude: t.PressureAltitudeFt,
    vertical_rate: t.VerticalRateFtMin,
    *,
    airspeed: t.TasKt | None = None,
    mass: t.MassKg | None = None,
    elapsed_time: t.DurationS | None = None,
    altitude_rate: t.VerticalRateFtS | None = None,
    groundspeed_rate: t.AccelerationKtS | None = None,
    airspeed_rate: t.AccelerationKtS | None = None,
    xp: ArrayApiNamespace,
):
    airspeed = groundspeed if airspeed is None else airspeed
    zeros = xp.zeros_like(groundspeed)

    normalised_mass = (
        zeros - 1
        if mass is None
        else (mass - aircraft.oew) / (aircraft.max_tow - aircraft.oew)
    )

    if elapsed_time is None:
        altitude_rate = (
            vertical_rate / 60 if altitude_rate is None else altitude_rate
        )
        groundspeed_rate = (
            zeros if groundspeed_rate is None else groundspeed_rate
        )
        airspeed_rate = zeros if airspeed_rate is None else airspeed_rate
    else:
        delta_time = _diff_bfill(elapsed_time, xp=xp)
        altitude_rate = (
            _diff_bfill(altitude, xp=xp) / delta_time
            if altitude_rate is None
            else altitude_rate
        )
        groundspeed_rate = (
            _diff_bfill(groundspeed, xp=xp) / delta_time
            if groundspeed_rate is None
            else groundspeed_rate
        )
        airspeed_rate = (
            _diff_bfill(airspeed, xp=xp) / delta_time
            if airspeed_rate is None
            else airspeed_rate
        )

    return xp.stack(
        (
            zeros + int(aircraft.engine_type),
            altitude_rate,
            groundspeed_rate,
            airspeed_rate,
            zeros + aircraft.wing_area,
            zeros + aircraft.max_alt,
            zeros + aircraft.max_tas,
            altitude,
            groundspeed,
            airspeed,
            vertical_rate,
            normalised_mass,
        ),
        axis=-1,
    )


def standardise(
    standardisation: AcropoleStandardisation,
    features,
):
    return (features - standardisation.minimums) / (
        standardisation.maximums - standardisation.minimums
    )


def predict_standardised(
    weights: AcropoleWeights,
    features_standardised,
    *,
    xp: ArrayApiNamespace,
):
    """Return the raw normalised fuel-flow prediction."""
    hidden = features_standardised
    for weight, bias in zip(weights.weights[:-1], weights.biases[:-1]):
        activation = hidden @ weight + bias
        hidden = xp.maximum(activation, xp.zeros_like(activation))

    logits = hidden @ weights.weights[-1] + weights.biases[-1]
    one = xp.ones_like(logits[..., 0])
    # very negative logits may overflow exp; the sigmoid still converges to 0.
    return one / (one + xp.exp(-logits[..., 0]))
