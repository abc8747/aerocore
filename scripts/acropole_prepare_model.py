#!/usr/bin/env -S uv run --no-default-groups --extra cli --extra httpx --extra onnx
# ruff: noqa: E501
"""Build redistributable Acropole runtime data artifacts.

Requires extras:

- `cli` for Typer
- `httpx` for downloading the upstream release
- `onnx` for reading the upstream ONNX model

Requires:

- network access to download the upstream release
"""

from __future__ import annotations

import csv
import io
import tarfile
from pathlib import Path
from typing import Any, BinaryIO, cast

import httpx
import onnx
import typer
from onnx import numpy_helper

import numpy as np
from aerocore.acropole import (
    ACROPOLE_AIRCRAFT_FILENAME,
    ACROPOLE_LICENSE_FILENAME,
    ACROPOLE_MODEL_FILENAME,
)

UPSTREAM_VERSION = "0.1.0"
UPSTREAM_URL = (
    "https://github.com/DGAC/Acropole/archive/refs/tags/v0.1.0.tar.gz"
)
_ARCHIVE_ROOT = f"Acropole-{UPSTREAM_VERSION}"
_AIRCRAFT_MEMBER = f"{_ARCHIVE_ROOT}/src/acropole/data/aircraft_params.csv"
_MODEL_MEMBER = f"{_ARCHIVE_ROOT}/src/acropole/models/acropole_fuel_model.onnx"
_LICENSE_MEMBER = f"{_ARCHIVE_ROOT}/LICENSE"

_FEATURE_MINIMUMS = np.array(
    [0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0],
    dtype=np.float32,
)
_FEATURE_MAXIMUMS = np.array(
    [1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1],
    dtype=np.float32,
)
_DENSE_LAYERS = (
    "dense_1",
    "dense_1_2",
    "dense_2_1",
    "dense_3_1",
    "dense_4_1",
)

_AIRCRAFT_COLUMNS = (
    "ACFT_ICAO_TYPE",
    "ENGINE_ICAO",
    "TRAIN_ON",
    "ENGINE_TYPE",
    "SURFACE",
    "MAX_OPE_ALTI",
    "MAX_OPE_SPEED",
    "OPE_EMPTY_WEIGHT",
    "MAX_TO_WEIGHT",
    "FUEL_FLOW_TO",
    "ENGINE_NUM",
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
)


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    return cast(BinaryIO, archive.extractfile(name)).read()


def _source_members(source_url: str) -> tuple[bytes, bytes, bytes]:
    response = httpx.get(source_url, follow_redirects=True)
    response.raise_for_status()
    with tarfile.open(
        fileobj=io.BytesIO(response.content),
        mode="r:gz",
    ) as archive:
        return (
            _read_member(archive, _MODEL_MEMBER),
            _read_member(archive, _AIRCRAFT_MEMBER),
            _read_member(archive, _LICENSE_MEMBER),
        )


def _convert_model(model_data: bytes, output: Path) -> None:
    model = onnx.load_model_from_string(model_data)
    tensors = {
        tensor.name: np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
        for tensor in model.graph.initializer
    }
    arrays: dict[str, Any] = {
        "feature_minimums": _FEATURE_MINIMUMS,
        "feature_maximums": _FEATURE_MAXIMUMS,
    }
    for index, layer in enumerate(_DENSE_LAYERS):
        prefix = f"StatefulPartitionedCall/sequential_1/{layer}"
        arrays[f"weight_{index}"] = tensors[f"{prefix}/Cast/ReadVariableOp:0"]
        arrays[f"bias_{index}"] = tensors[f"{prefix}/Add/ReadVariableOp:0"]
    np.savez(output, **arrays)
    output.chmod(0o644)


def _convert_aircraft(data: bytes, output: Path) -> None:
    rows = csv.DictReader(io.StringIO(data.decode("utf-8")))
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_AIRCRAFT_COLUMNS)
        writer.writeheader()
        for row in rows:
            converted = {name: row[name] for name in _AIRCRAFT_COLUMNS}
            for name in ("TRAIN_ON", "ENGINE_TYPE", "ENGINE_NUM"):
                converted[name] = str(int(float(converted[name])))
            for name in (
                "SURFACE",
                "MAX_OPE_ALTI",
                "MAX_OPE_SPEED",
                "OPE_EMPTY_WEIGHT",
                "MAX_TO_WEIGHT",
                "FUEL_FLOW_TO",
            ):
                converted[name] = format(float(converted[name]), ".15g")
            writer.writerow(converted)
    output.chmod(0o644)


@app.command()
def build(
    output_dir: Path = Path("dist/acropole-v0.1.0"),
    source_url: str = UPSTREAM_URL,
) -> None:
    """Download Acropole and build the release artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_data, aircraft_data, license_data = _source_members(source_url)

    model_output = output_dir / ACROPOLE_MODEL_FILENAME
    aircraft_output = output_dir / ACROPOLE_AIRCRAFT_FILENAME
    license_output = output_dir / ACROPOLE_LICENSE_FILENAME
    _convert_model(model_data, model_output)
    _convert_aircraft(aircraft_data, aircraft_output)
    license_output.write_bytes(license_data)
    license_output.chmod(0o644)

    typer.echo(model_output)
    typer.echo(aircraft_output)
    typer.echo(license_output)


if __name__ == "__main__":
    app()
