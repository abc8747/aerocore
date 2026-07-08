# avoid importing anything other than stdlib, use local imports instead.

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypeAlias

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
DateArg: TypeAlias = Annotated[
    datetime,
    typer.Argument(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]),
]
OptionalDateArg: TypeAlias = Annotated[
    datetime | None,
    typer.Argument(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]),
]


@app.command()
def dirs() -> None:
    """Shows directories"""
    from .utils import default_cache_dir

    stderr.print(f"default_cache_dir={default_cache_dir()}")


#
# data
#


@app.command()
def data_acropole_sync(
    cache_dir: Path | None = None,
    force: bool = False,
) -> None:
    """Download prepared Acropole model data."""
    from .acropole import sync_data

    for path in asyncio.run(sync_data(cache_dir=cache_dir, force=force)):
        logger.info("synced=%s", path)


@app.command()
def data_adsblol_metadata_sync(
    repo_output_specs: list[str] | None = None,
) -> None:
    """Sync adsb.lol github release metadata to JSONL"""
    from .data.adsblol import (
        DEFAULT_GH_API_BASE_URL,
        fp_datalinks_jsonl,
        fp_modes_adsb_jsonl,
        get_github_auth_token,
        run_metadata,
    )
    from .utils import default_cache_dir

    dir_cache = default_cache_dir()
    modes_path = fp_modes_adsb_jsonl(dir_cache)
    datalinks_path = fp_datalinks_jsonl(dir_cache)

    specs = repo_output_specs or [
        f"adsblol/globe_history_2023={modes_path}",
        f"adsblol/globe_history_2024={modes_path}",
        f"adsblol/globe_history_2025={modes_path}",
        f"adsblol/globe_history_2026={modes_path}",
        f"adsblol/aircraft-data-links-2025={datalinks_path}",
        f"adsblol/aircraft-data-links-2026={datalinks_path}",
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

    result = asyncio.run(
        run_metadata(
            grouped_specs=grouped_specs,
            base_url=DEFAULT_GH_API_BASE_URL,
            auth_token=get_github_auth_token(),
        )
    )
    for file_result in result.files:
        logger.info(
            "metadata=%s existing=%s appended=%s",
            file_result.path,
            file_result.existing_assets,
            file_result.appended_assets,
        )
        for repository_result in file_result.repositories:
            logger.info(
                "%s pages=%s seen=%s appended=%s stopped_known=%s "
                "hit_page_limit=%s",
                repository_result.repository,
                repository_result.pages_fetched,
                repository_result.assets_seen,
                repository_result.assets_appended,
                repository_result.stopped_on_known_page,
                repository_result.hit_page_limit,
            )


@app.command()
def data_adsblol_metadata_plot(
    paths: list[Path] | None = None,
    output: Path = Path("/tmp/adsblol_metadata.png"),
) -> None:
    """Plot adsb.lol metadata raw hosted bytes over time"""
    from .data.adsblol import (
        build_adsblol_figure,
        load_metadata_daily_sizes,
        load_modes_daily_sizes,
    )
    from .utils import default_cache_dir

    input_paths = paths or [default_cache_dir()]
    datalinks_sizes = load_metadata_daily_sizes(
        input_paths,
        "adsblol_datalinks",
    )
    modes_sizes = load_modes_daily_sizes(input_paths, "adsblol_modes_adsb")
    if not (datalinks_sizes or modes_sizes):
        raise typer.Exit(code=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_adsblol_figure(modes_sizes, datalinks_sizes)
    fig.savefig(output, dpi=180)
    logger.info(
        "wrote %s modes_days=%s datalinks_days=%s",
        output,
        len(modes_sizes),
        len(datalinks_sizes),
    )


def _datalink_plan_lines(result: object) -> list[str]:
    from .data.adsblol import DatalinkDownloadResult

    typed_result = result
    assert isinstance(typed_result, DatalinkDownloadResult)
    plan = typed_result.plan
    lines = [
        f"metadata={plan.metadata_path}",
        f"output_root={plan.output_root}",
        (
            f"selected={plan.selected} existing={plan.existing} "
            f"missing={plan.missing}"
        ),
    ]
    for repository, selected in plan.selected_by_repo.items():
        existing = plan.existing_by_repo.get(repository, 0)
        lines.append(
            f"{repository} selected={selected} existing={existing} "
            f"missing={selected - existing}"
        )
    return lines


@app.command()
def data_adsblol_datalinks_download(
    start_date: OptionalDateArg = None,
    end_date: OptionalDateArg = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    jobs: int | None = None,
) -> None:
    """Plan and run the adsb.lol datalinks JSONL download pipeline"""
    from .data.adsblol import (
        datalinks_download,
        dir_datalinks,
        fp_datalinks_jsonl,
    )
    from .utils import default_cache_dir

    dir_cache = default_cache_dir()
    metadata_path = fp_datalinks_jsonl(dir_cache).expanduser()
    output_root = (
        output_dir.expanduser()
        if output_dir is not None
        else dir_datalinks(dir_cache)
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            "metadata file not found: "
            f"{metadata_path}. run metadata sync first."
        )

    if (
        start_date is not None
        and end_date is not None
        and start_date.date() > end_date.date()
    ):
        raise ValueError("start_date must be on or before end_date")

    result = datalinks_download(
        metadata_path=metadata_path,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
        jobs=jobs,
    )
    lines = _datalink_plan_lines(result)
    if dry_run:
        print("\n".join(lines))
        if result.plan.missing == 0:
            print("nothing to download")
        return

    for line in lines:
        logger.info(line)
    if result.plan.missing == 0:
        logger.info("nothing to download")
    else:
        logger.info(
            "done existing=%s downloaded=%s unavailable=%s",
            result.existing,
            result.downloaded,
            result.unavailable,
        )


if __name__ == "__main__":
    app()
