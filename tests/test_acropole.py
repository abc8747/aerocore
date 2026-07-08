# mypy: disable-error-code="no-untyped-def"
from typing import Any, NamedTuple

import pytest

import numpy as np
from aerocore.acropole import (
    AcropoleAircraftDatabase,
    AcropoleEngineType,
    AcropoleModel,
    fuel_flow,
    input_features,
    load_aircraft_database,
    load_model,
)
from aerocore.utils import tree_map


class Backend(NamedTuple):
    xp: Any
    model: AcropoleModel[Any]


@pytest.fixture(scope="module")
def database(acropole_data_paths) -> AcropoleAircraftDatabase:
    return load_aircraft_database(acropole_data_paths[1])


@pytest.fixture(scope="module")
def model(acropole_data_paths) -> AcropoleModel[Any]:
    return load_model(acropole_data_paths[0])


@pytest.fixture(scope="module", params=("numpy", "jax", "torch"))
def backend(
    request: pytest.FixtureRequest,
    model: AcropoleModel[Any],
) -> Backend:
    if request.param == "numpy":
        xp = np
    elif request.param == "torch":
        xp = pytest.importorskip("torch")
    else:
        xp = pytest.importorskip("jax.numpy")
    return Backend(
        xp,
        tree_map(lambda array: xp.asarray(array, dtype=xp.float32), model),
    )


def test_load_aircraft_database(database: AcropoleAircraftDatabase) -> None:
    assert database["A320"].representative_engine == "CFM56-5B4/P"
    assert database["A320"].trained is True
    assert database["A320"].engine_type is AcropoleEngineType.JET
    assert database["A320"].engine_count == 2
    assert database["A319"].trained is False
    assert database["AT76"].engine_type is AcropoleEngineType.TURBOPROP


def test_input_features(database: AcropoleAircraftDatabase) -> None:
    features = input_features(
        database["A320"],
        np.asarray([400.0, 408.0, 420.0]),
        np.asarray([1000.0, 1012.0, 1028.0]),
        np.asarray([180.0, 180.0, 180.0]),
        airspeed=np.asarray([410.0, 416.0, 428.0]),
        mass=np.asarray([60000.0, 60000.0, 60000.0]),
        elapsed_time=np.asarray([0.0, 4.0, 8.0]),
        xp=np,
    )
    np.testing.assert_allclose(features[:, 1], [3.0, 3.0, 4.0])
    np.testing.assert_allclose(features[:, 2], [2.0, 2.0, 3.0])
    np.testing.assert_allclose(features[:, 3], [1.5, 1.5, 3.0])
    np.testing.assert_allclose(features[:, 7], [1000.0, 1012.0, 1028.0])
    np.testing.assert_allclose(features[:, 8], [400.0, 408.0, 420.0])


def test_fuel_flow(
    backend: Backend,
    database: AcropoleAircraftDatabase,
) -> None:
    xp, model = backend
    actual = fuel_flow(
        model=model,
        aircraft=database["A320"],
        groundspeed=xp.asarray([180.0, 450.0, 250.0], dtype=xp.float32),
        altitude=xp.asarray([0.0, 30_000.0, 40_000.0], dtype=xp.float32),
        vertical_rate=xp.asarray([3_000.0, 0.0, -2_000.0], dtype=xp.float32),
        xp=xp,
    )
    np.testing.assert_allclose(
        np.asarray(actual),
        [1.91229012, 0.75688555, 0.09444639],
        rtol=1e-3,
        atol=1e-6,
    )
