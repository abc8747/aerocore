# avoid importing anything other than stdlib, use local imports instead.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

stderr = Console(stderr=True)
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=stderr)],
)
logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)


@app.command()
def dirs() -> None:
    """Shows directories"""
    from .utils import default_cache_dir

    stderr.print(f"default_cache_dir={default_cache_dir()}")


#
# data
#


@app.command()
def data_adsblol_metadata_sync(
    repo_output_specs: list[str] | None = None,
) -> None:
    """Syncs github release metadata to JSONL."""
    from .data.adsblol import (
        DEFAULT_GH_API_BASE_URL,
        get_github_auth_token,
        metadata_async,
    )
    from .utils import default_cache_dir

    dir_cache = default_cache_dir()
    dir_modes = dir_cache / "adsblol_modes_adsb.jsonl"
    dir_datalinks = dir_cache / "adsblol_datalinks.jsonl"

    specs = repo_output_specs or [
        f"adsblol/globe_history_2023={dir_modes}",
        f"adsblol/globe_history_2024={dir_modes}",
        f"adsblol/globe_history_2025={dir_modes}",
        f"adsblol/globe_history_2026={dir_modes}",
        f"adsblol/aircraft-data-links-2025={dir_datalinks}",
        f"adsblol/aircraft-data-links-2026={dir_datalinks}",
    ]
    grouped_specs: dict[Path, list[str]] = {}
    for spec in specs:
        if "=" not in spec:
            raise typer.BadParameter("expected REPO=FILE")
        repository, output = spec.split("=", 1)
        if not (repository := repository.strip()):
            raise typer.BadParameter("repository name cannot be empty")
        if not (output_text := output.strip()):
            raise typer.BadParameter("output path cannot be empty")
        output_path = Path(output_text).expanduser()
        grouped_specs.setdefault(output_path, []).append(repository)

    asyncio.run(
        metadata_async(
            grouped_specs=grouped_specs,
            base_url=DEFAULT_GH_API_BASE_URL,
            auth_token=get_github_auth_token(),
        )
    )
