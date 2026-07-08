# aerocore

[![image](https://img.shields.io/pypi/v/aerocore.svg)](https://pypi.python.org/pypi/aerocore)
[![image](https://img.shields.io/pypi/l/aerocore.svg)](https://pypi.python.org/pypi/aerocore)
[![image](https://img.shields.io/pypi/pyversions/aerocore.svg)](https://pypi.python.org/pypi/aerocore)
[![image](https://img.shields.io/pypi/status/aerocore)](https://pypi.python.org/pypi/aerocore)

`aerocore` is a lightweight toolbox for air traffic management research. It only has two dependencies: [`numpy`](https://numpy.org/) and [`isqx` (a units library)](https://github.com/abc8747/isqx). Additional features are available when optional dependencies are installed.

It supports multiple numerical backends through the [Array API](https://data-apis.org/array-api), including JAX arrays and PyTorch tensors.

## Installation

`aerocore` is currently under heavy development and **not considered stable**. For the latest version:

```sh
# with pip
pip install aerocore
# with uv
uv add aerocore
```

Depending on your use case, you can pick the optional dependencies you need:

- `polars`: support for [polars](https://github.com/pola-rs/polars) and its Array API shim (postprocessing third party data)
- `httpx`: support for [httpx](https://github.com/encode/httpx) (downloading data from external sources)
- `xarray`: support for [xarray](https://github.com/pydata/xarray) (ARCO-ERA5 weather grids, working with NetCDF)
- `jax`: support for [JAX](https://github.com/jax-ml/jax) (automatic differentiation/GPU acceleration support)
- `matplotlib`: plotting
- `platformdirs`: reading/writing cache/config files
- `cli`: command line scripts
- `all`: install all optional dependencies (not recommended!)

For example:

```sh
pip install "aerocore[httpx,polars,cli]"
```

## Usage

For the CLI:

```sh
uv run aerocore --help
```

## Development

```sh
git clone https://github.com/abc8747/aerocore --depth=1
cd aerocore
uv venv
# standard
uv sync --all-groups --extra all
# if you want to benchmark with GPU support 
uv sync --all-groups --all-extras --no-extra 'torch-cpu'
```

To run scripts:

```sh
uv run examples/capabilities_jax.py
```

For documentation:

```sh
uv run zensical serve
```

It should then host at <http://127.0.0.1:8000/aerocore/>.

For testing:

```sh
# by default, it tests everything
uv run pytest
# skip slow network tests that fetch external data (e.g. adsb.lol, icao.int...)
uv run pytest -m "not network"
```

### Contributing

PRs or issues are very welcome!

We use [Ruff](https://github.com/astral-sh/ruff) for linting and [MyPy](https://github.com/python/mypy) for type checking. Locally, run the following before committing:

```sh
just fmt
just check
```

License: MIT
