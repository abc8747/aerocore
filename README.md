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

- `all`: install all optional dependencies
- `polars`: support for [polars](https://github.com/pola-rs/polars) and its Array API shim (postprocessing third party data)
- `httpx`: support for [httpx](https://github.com/encode/httpx) (downloading data from external sources)
- `xarray`: support for [xarray](https://github.com/pydata/xarray) (ARCO-ERA5 weather grids, working with NetCDF)
- `jax`: support for [JAX](https://github.com/jax-ml/jax) (automatic differentiation support)
- `matplotlib`: plotting

For example:

```sh
pip install "aerocore[httpx,polars]"
```

## Development

```sh
git clone https://github.com/abc8747/aerocore --depth=1
cd aerocore
uv venv
uv sync --all-extras --all-groups
```

To run scripts:

```sh
uv run examples/autodiff.py
```

Alternatively, activate your virtualenv:

```sh
source .venv/bin/activate
python3 examples/autodiff.py
```

### Documentation

```sh
uv run mkdocs serve
```

Then, navigate to <http://127.0.0.1:8000/aerocore/>.

### Contributing

PRs or issues are very welcome!

We use [Ruff](https://github.com/astral-sh/ruff) for linting and [MyPy](https://github.com/python/mypy) for type checking. Locally, run the following before committing:

```sh
just fmt
just check
```

License: MIT
