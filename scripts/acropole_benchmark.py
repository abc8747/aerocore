#!/usr/bin/env -S uv run --no-default-groups --extra cli --extra jax_gpu --extra matplotlib --extra onnx --extra onnx_gpu --extra torch_gpu
# ruff: noqa: E501
# mypy: disable-error-code="type-arg"
"""Benchmark Array API functions across numerical backends.

The harness is written to benchmark other functions in the future. It currently
runs the Acropole workload defined at the end of this script.

Requires extras:

- `cli` for Typer
- `matplotlib` for plotting
- `jax_gpu` for JAX CPU and GPU
- `torch_gpu` for CUDA-enabled PyTorch
- `onnx` for exporting the benchmark function to ONNX
- `onnx_gpu` for ONNX Runtime CPU and GPU
- `onnxscript` for Dynamo-based Torch->ONNX export (default in pytorch>=2.9)

Requires:

- an NVIDIA GPU and compatible driver for GPU targets
"""

from __future__ import annotations

import io
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypedDict, cast

import typer
from typing_extensions import assert_never

import aerocore.types as t
import numpy as np
from aerocore.acropole import (
    ACROPOLE_MODEL_FILENAME,
    AcropoleModel,
    data_dir,
    load_model,
    predict_standardised,
    standardise,
)
from aerocore.utils import tree_map

Array = Any
ArrayApiFunction = Callable[..., Array]
Call = Callable[[], Array]
Convert = Callable[[np.ndarray], Array]


class PreparedArguments(NamedTuple):
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


Prepare = Callable[[Convert], PreparedArguments]


def no_prepare(convert: Convert) -> PreparedArguments:
    return PreparedArguments((), {})


@dataclass(frozen=True)
class Workload:
    """One invocation of a function accepting an `xp` keyword."""

    name: str
    function: ArrayApiFunction
    inputs: tuple[np.ndarray, ...]
    batch_size: int
    prepare: Prepare = no_prepare

    def bind(self, xp: Any, convert: Convert) -> Callable[..., Array]:
        prepared = self.prepare(convert)

        def invoke(*inputs: Array) -> Array:
            return self.function(
                *prepared.args,
                *inputs,
                xp=xp,
                **prepared.kwargs,
            )

        return invoke

    def reference(self) -> np.ndarray:
        invoke = self.bind(np, np.asarray)
        return np.asarray(invoke(*self.inputs))


@dataclass(frozen=True)
class Backend:
    name: str
    device: str
    version: str
    # timed calls keep model state and inputs resident on the device
    # host transfers are performed only for correctness checking
    call: Call
    synchronize: Callable[[Array], None]
    to_host: Callable[[Array], np.ndarray]


def noop_synchronize(output: Array) -> None:
    pass


def numpy_backend(workload: Workload) -> Backend:
    invoke = workload.bind(np, np.asarray)
    return Backend(
        name="numpy",
        device="cpu",
        version=np.__version__,
        call=lambda: invoke(*workload.inputs),
        synchronize=noop_synchronize,
        to_host=np.asarray,
    )


def jax_backend(
    workload: Workload,
    device_name: str,
    *,
    compiled: bool,
) -> Backend:
    import jax
    import jax.numpy as jnp

    target = jax.devices(device_name)[0]

    def convert(array: np.ndarray) -> Array:
        return jnp.asarray(array, device=target)

    invoke = workload.bind(jnp, convert)
    if compiled:
        invoke = jax.jit(invoke, device=target)
    inputs = tuple(convert(array) for array in workload.inputs)

    def synchronize(output: Array) -> None:
        output.block_until_ready()

    return Backend(
        name="jax-jit" if compiled else "jax-eager",
        device=device_name,
        version=jax.__version__,
        call=lambda: invoke(*inputs),
        synchronize=synchronize,
        to_host=np.asarray,
    )


def torch_backend(
    workload: Workload,
    device_name: str,
    *,
    compiled: bool,
) -> Backend:
    import torch

    target = torch.device("cuda" if device_name == "gpu" else "cpu")
    if device_name == "gpu":
        torch.set_float32_matmul_precision("high")

    def convert(array: np.ndarray) -> Array:
        return torch.from_numpy(array).to(target)

    invoke = workload.bind(torch, convert)
    if compiled:
        invoke = torch.compile(invoke, mode="default")
    inputs = tuple(convert(array) for array in workload.inputs)

    def resident() -> Array:
        with torch.inference_mode():
            return invoke(*inputs)

    def synchronize(output: Array) -> None:
        if device_name == "gpu":
            torch.cuda.synchronize(target)

    def to_host(output: Array) -> np.ndarray:
        return output.detach().cpu().numpy()  # type: ignore[no-any-return]

    return Backend(
        name="torch-compile" if compiled else "torch-eager",
        device=device_name,
        version=torch.__version__,
        call=resident,
        synchronize=synchronize,
        to_host=to_host,
    )


def onnxruntime_backend(
    workload: Workload,
    reference: np.ndarray,
    device_name: str,
) -> Backend:
    import onnxruntime as ort
    import torch

    invoke = workload.bind(torch, torch.from_numpy)
    example_inputs = tuple(torch.from_numpy(array) for array in workload.inputs)
    input_names = [f"input_{index}" for index in range(len(example_inputs))]

    class Module(torch.nn.Module):
        def forward(self, *inputs: Array) -> Array:
            return invoke(*inputs)

    batch = torch.export.Dim("batch")
    dynamic_shapes = {
        "inputs": tuple(
            {0: batch} if array.ndim > 0 else None for array in workload.inputs
        )
    }

    module = Module().eval()
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        example_inputs,
        cast(Any, buffer),
        input_names=input_names,
        output_names=["output"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
    )
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device_name == "gpu"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(buffer.getvalue(), providers=providers)
    feed = dict(zip(input_names, workload.inputs))

    if device_name == "cpu":
        return Backend(
            name="onnxruntime",
            device="cpu",
            version=ort.__version__,
            call=lambda: session.run(None, feed)[0],
            synchronize=noop_synchronize,
            to_host=np.asarray,
        )

    binding = session.io_binding()
    for name, array in feed.items():
        value = ort.OrtValue.ortvalue_from_numpy(array, "cuda", 0)
        binding.bind_ortvalue_input(name, value)
    output = ort.OrtValue.ortvalue_from_shape_and_type(
        reference.shape,
        reference.dtype,
        "cuda",
        0,
    )
    binding.bind_ortvalue_output("output", output)

    def resident() -> Array:
        session.run_with_iobinding(binding)
        return output

    return Backend(
        name="onnxruntime",
        device="gpu",
        version=ort.__version__,
        call=resident,
        synchronize=lambda value: binding.synchronize_outputs(),
        to_host=lambda value: value.numpy(),
    )


class BenchmarkResult(TypedDict):
    workload: str
    backend: str
    device: str
    batch_size: int
    setup_ms: float
    first_call_ms: float
    median_us: float
    p05_us: float
    p95_us: float
    samples_per_second: float
    iterations: int
    max_abs_error: float
    max_rel_error: float
    backend_version: str


class Target(str, Enum):
    NUMPY_CPU = "numpy-cpu"
    JAX_EAGER_CPU = "jax-eager-cpu"
    JAX_EAGER_GPU = "jax-eager-gpu"
    JAX_JIT_CPU = "jax-jit-cpu"
    JAX_JIT_GPU = "jax-jit-gpu"
    TORCH_EAGER_CPU = "torch-eager-cpu"
    TORCH_EAGER_GPU = "torch-eager-gpu"
    TORCH_COMPILE_CPU = "torch-compile-cpu"
    TORCH_COMPILE_GPU = "torch-compile-gpu"
    ONNXRUNTIME_CPU = "onnxruntime-cpu"
    ONNXRUNTIME_GPU = "onnxruntime-gpu"


def make_backend(
    name: Target,
    workload: Workload,
    reference: np.ndarray,
) -> Backend:
    match name:
        case Target.NUMPY_CPU:
            return numpy_backend(workload)
        case Target.JAX_EAGER_CPU:
            return jax_backend(workload, "cpu", compiled=False)
        case Target.JAX_EAGER_GPU:
            return jax_backend(workload, "gpu", compiled=False)
        case Target.JAX_JIT_CPU:
            return jax_backend(workload, "cpu", compiled=True)
        case Target.JAX_JIT_GPU:
            return jax_backend(workload, "gpu", compiled=True)
        case Target.TORCH_EAGER_CPU:
            return torch_backend(workload, "cpu", compiled=False)
        case Target.TORCH_EAGER_GPU:
            return torch_backend(workload, "gpu", compiled=False)
        case Target.TORCH_COMPILE_CPU:
            return torch_backend(workload, "cpu", compiled=True)
        case Target.TORCH_COMPILE_GPU:
            return torch_backend(workload, "gpu", compiled=True)
        case Target.ONNXRUNTIME_CPU:
            return onnxruntime_backend(workload, reference, "cpu")
        case Target.ONNXRUNTIME_GPU:
            return onnxruntime_backend(workload, reference, "gpu")
    assert_never(name)


class ErrorStats(NamedTuple):
    max_abs_error: float
    max_rel_error: float


def relative_error(
    reference: np.ndarray,
    actual: np.ndarray,
) -> ErrorStats:
    difference = np.abs(reference - actual)
    denominator = np.maximum(np.abs(reference), np.float32(1e-7))
    return ErrorStats(
        float(difference.max()),
        float((difference / denominator).max()),
    )


@dataclass(frozen=True)
class Measurement:
    iterations: int
    samples_us: list[float]

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us)

    @property
    def p05_us(self) -> float:
        return statistics.quantiles(
            self.samples_us,
            n=20,
            method="inclusive",
        )[0]

    @property
    def p95_us(self) -> float:
        return statistics.quantiles(
            self.samples_us,
            n=20,
            method="inclusive",
        )[18]


@dataclass(frozen=True)
class Criterion:
    warm_up: t.DurationS[float] = 0.1
    measurement: t.DurationS[float] = 0.15
    sample_count: int = 10

    def measure(
        self,
        call: Call,
        synchronize: Callable[[Array], None],
    ) -> Measurement:
        warm_up_end = time.perf_counter() + self.warm_up
        while time.perf_counter() < warm_up_end:
            output = call()
            synchronize(output)

        start = time.perf_counter_ns()
        output = call()
        synchronize(output)
        elapsed_ns = max(time.perf_counter_ns() - start, 1)
        sample_seconds = self.measurement / self.sample_count
        iterations = max(1, math.ceil(sample_seconds * 1e9 / elapsed_ns))

        samples = []
        for _ in range(self.sample_count):
            start = time.perf_counter_ns()
            for _ in range(iterations):
                output = call()
                synchronize(output)
            duration = time.perf_counter_ns() - start
            samples.append(duration / iterations / 1000)
        return Measurement(iterations, samples)


def benchmark(
    workload: Workload,
    target_name: Target,
    criterion: Criterion,
) -> BenchmarkResult:
    reference = workload.reference()
    setup_start = time.perf_counter_ns()
    backend = make_backend(target_name, workload, reference)
    setup_ms = (time.perf_counter_ns() - setup_start) / 1e6

    first_start = time.perf_counter_ns()
    first_output = backend.call()
    backend.synchronize(first_output)
    first_call_ms = (time.perf_counter_ns() - first_start) / 1e6
    error = relative_error(reference, backend.to_host(first_output))
    measurement = criterion.measure(backend.call, backend.synchronize)
    return BenchmarkResult(
        workload=workload.name,
        backend=backend.name,
        device=backend.device,
        batch_size=workload.batch_size,
        setup_ms=setup_ms,
        first_call_ms=first_call_ms,
        median_us=measurement.median_us,
        p05_us=measurement.p05_us,
        p95_us=measurement.p95_us,
        samples_per_second=(workload.batch_size * 1e6 / measurement.median_us),
        iterations=measurement.iterations,
        max_abs_error=error.max_abs_error,
        max_rel_error=error.max_rel_error,
        backend_version=backend.version,
    )


def plot_results(
    results: Iterable[BenchmarkResult],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    series: defaultdict[tuple[str, str], list[BenchmarkResult]] = defaultdict(
        list
    )
    backend_versions: defaultdict[str, set[str]] = defaultdict(set)
    batch_sizes = set()
    for result in results:
        series[(result["backend"], result["device"])].append(result)
        backend_versions[result["backend"]].add(result["backend_version"])
        batch_sizes.add(result["batch_size"])

    backends = list(dict.fromkeys(backend for backend, _ in series))
    colors = plt.colormaps["tab10"]
    backend_colors = {
        backend: colors(index % colors.N)
        for index, backend in enumerate(backends)
    }
    device_styles = {
        "cpu": ("o", "-"),
        "gpu": ("x", "--"),
    }

    plt.style.use("dark_background")

    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    layout = figure.add_gridspec(
        2,
        2,
        height_ratios=(5, 1),
        hspace=0.05,
    )
    axes = figure.add_subplot(layout[0, :])
    backend_axes = figure.add_subplot(layout[1, 0])
    device_axes = figure.add_subplot(layout[1, 1])
    backend_axes.axis("off")
    device_axes.axis("off")

    for (backend, device), values in series.items():
        values.sort(key=lambda value: value["batch_size"])
        marker, linestyle = device_styles[device]
        axes.errorbar(
            [value["batch_size"] for value in values],
            medians := [value["median_us"] for value in values],
            yerr=[
                [
                    median - value["p05_us"]
                    for median, value in zip(medians, values)
                ],
                [
                    value["p95_us"] - median
                    for median, value in zip(medians, values)
                ],
            ],
            color=backend_colors[backend],
            marker=marker,
            linestyle=linestyle,
            markersize=3,
            markeredgewidth=0.75,
            capsize=2,
        )

    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    axes.set_xticks(
        ticks := sorted(batch_sizes),
        [f"{value:,}" for value in ticks],
        rotation=30,
        ha="right",
    )
    axes.set_xlabel("batch size")
    axes.set_ylabel("median latency per batch call (µs)")
    axes.grid(True, which="both", linewidth=0.5, alpha=0.35)

    backend_axes.legend(
        handles=[
            Line2D(
                [],
                [],
                color=backend_colors[backend],
                label=(
                    f"{backend} {', '.join(sorted(backend_versions[backend]))}"
                ),
            )
            for backend in backends
        ],
        title="backend",
        loc="upper center",
        frameon=True,
    )
    device_axes.legend(
        handles=[
            Line2D(
                [],
                [],
                color=plt.rcParams["text.color"],
                marker=marker,
                linestyle=linestyle,
                markersize=3,
                markeredgewidth=0.75,
                label=device,
            )
            for device, (marker, linestyle) in device_styles.items()
        ],
        title="device",
        loc="upper center",
        frameon=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def write_results(
    results: Iterable[BenchmarkResult],
    output: Path | None,
) -> None:
    text = "".join(
        json.dumps(result, sort_keys=True) + "\n" for result in results
    )
    if output is None:
        typer.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


#
# acropole workload
#


def infer(model: AcropoleModel[Any], features: Array, *, xp: Any) -> Array:
    standardised = standardise(model.standardisation, features)
    return predict_standardised(model.weights, standardised, xp=xp)


def acropole_workload(model: AcropoleModel[Any], batch_size: int) -> Workload:
    rng = np.random.default_rng(20260708 + batch_size)
    unit = rng.uniform(-0.1, 1.1, size=(batch_size, 12)).astype(np.float32)
    minimums = model.standardisation.minimums
    features = minimums + unit * (model.standardisation.maximums - minimums)
    return Workload(
        name="Acropole",
        function=infer,
        inputs=(features,),
        batch_size=batch_size,
        prepare=lambda convert: PreparedArguments(
            (tree_map(convert, model),), {}
        ),
    )


app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    target: list[Target] = [
        Target.NUMPY_CPU,
        Target.JAX_EAGER_CPU,
        Target.JAX_EAGER_GPU,
        Target.JAX_JIT_CPU,
        Target.JAX_JIT_GPU,
        Target.TORCH_EAGER_CPU,
        Target.TORCH_EAGER_GPU,
        Target.TORCH_COMPILE_CPU,
        Target.TORCH_COMPILE_GPU,
        Target.ONNXRUNTIME_CPU,
        Target.ONNXRUNTIME_GPU,
    ],
    batch_size: list[int] = [4**i for i in range(11)],
    warm_up: t.DurationS[float] = 0.1,
    measurement: t.DurationS[float] = 0.15,
    sample_count: int = 10,
    model: Path = data_dir() / ACROPOLE_MODEL_FILENAME,
    output: Path | None = None,
    plot: Path | None = Path("docs/assets/img/acropole-benchmark.png"),
) -> None:
    """Benchmark Acropole across one or more backends."""
    acropole = load_model(model)
    criterion = Criterion(warm_up, measurement, sample_count)
    results = [
        benchmark(
            acropole_workload(acropole, size),
            target_name,
            criterion,
        )
        for target_name in target
        for size in batch_size
    ]
    write_results(results, output)
    if plot is not None:
        plot_results(results, plot)


if __name__ == "__main__":
    app()
