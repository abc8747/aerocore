"""adsb.lol Mode-S/ADS-B snapshots and aircraft data links.

Requires extras:

- `networking` downloading metadata
- `platformdirs` storing metadata to default cache

Requires:

- curl for downloading datalinks
- zstd for datalink decompression
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from itertools import accumulate
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from .. import types as t

if TYPE_CHECKING:
    from httpx import AsyncClient
    from matplotlib.figure import Figure

logger = getLogger(__name__)

#
# metadata: syncing
#


DEFAULT_GH_API_BASE_URL = "https://api.github.com"
MAX_RELEASES_PER_PAGE = 100


def fp_modes_adsb_jsonl(base_dir: Path) -> Path:
    return base_dir / "adsblol_modes_adsb.jsonl"


def fp_datalinks_jsonl(base_dir: Path) -> Path:
    return base_dir / "adsblol_datalinks.jsonl"


@dataclass(frozen=True, slots=True)
class AssetKey:
    repository: str
    asset_id: int


# NOTE: some gh fields are omitted for brevity


class GitHubReleaseAsset(TypedDict):
    id: int
    name: str
    size: int
    digest: str | None
    browser_download_url: str


class GitHubRelease(TypedDict):
    tag_name: str
    assets: list[GitHubReleaseAsset]


class ReleasePage(TypedDict):
    repository: str
    page: int
    per_page: int
    releases: list[GitHubRelease]


class AdsblolAsset(TypedDict):
    repository: str
    release_tag: str
    asset_id: int
    asset_name: str
    asset_size: int
    asset_digest: str | None
    browser_download_url: str


@dataclass(frozen=True, slots=True)
class RepositoryMetadataSyncResult:
    repository: str
    pages_fetched: int
    assets_seen: int
    assets_appended: int
    stopped_on_known_page: bool
    hit_page_limit: bool


@dataclass(frozen=True, slots=True)
class MetadataFileSyncResult:
    path: Path
    existing_assets: int
    appended_assets: int
    repositories: tuple[RepositoryMetadataSyncResult, ...]


@dataclass(frozen=True, slots=True)
class MetadataSyncResult:
    files: tuple[MetadataFileSyncResult, ...]


def get_github_auth_token() -> str | None:
    for env_name in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(env_name)
        if token is not None and token.strip():
            return token.strip()

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    token = result.stdout.strip()
    return token or None


def github_headers(auth_token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if auth_token is not None:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


async def fetch_release_page(
    client: AsyncClient,
    repository: str,
    *,
    page: int = 1,
    per_page: int = MAX_RELEASES_PER_PAGE,
) -> ReleasePage | None:
    response = await client.get(
        f"/repos/{repository}/releases",
        params={"page": page, "per_page": per_page},
    )
    if not response.is_success:
        return None
    releases: list[GitHubRelease] = response.json()
    return {
        "repository": repository,
        "page": page,
        "per_page": per_page,
        "releases": releases,
    }


def iter_release_asset_downloads(page: ReleasePage) -> Iterator[AdsblolAsset]:
    for release in page["releases"]:
        for asset in release["assets"]:
            yield {
                "repository": page["repository"],
                "release_tag": release["tag_name"],
                "asset_id": int(asset["id"]),
                "asset_name": str(asset["name"]),
                "asset_size": int(asset["size"]),
                "asset_digest": asset.get("digest"),
                "browser_download_url": str(asset["browser_download_url"]),
            }


def iter_metadata_rows(path: Path) -> Iterator[AdsblolAsset]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if stripped := line.strip():
                yield json.loads(stripped)


def _known_asset_keys(path: Path) -> set[AssetKey]:
    if not path.exists():
        return set()
    return {
        AssetKey(row["repository"], int(row["asset_id"]))
        for row in iter_metadata_rows(path)
    }


async def _sync_repository_incremental(
    client: AsyncClient,
    repository: str,
    known_asset_keys: set[AssetKey],
    *,
    per_page: int,
    max_pages: int | None,
) -> tuple[RepositoryMetadataSyncResult, list[AdsblolAsset]]:
    """Fetch newest-first release pages until a page has no new asset ids."""
    page_number = 1
    pages_fetched = 0
    assets_seen = 0
    appended: list[AdsblolAsset] = []
    stopped_on_known_page = False
    hit_page_limit = False

    while True:
        page = await fetch_release_page(
            client,
            repository=repository,
            page=page_number,
            per_page=per_page,
        )
        if page is None:
            logger.warning(
                "repo %s page %s got error, skipping", repository, page
            )
            page_number += 1
            continue
        pages_fetched += 1
        releases = page["releases"]
        if not releases:
            break

        page_appended = 0
        for row in iter_release_asset_downloads(page):
            assets_seen += 1
            key = AssetKey(row["repository"], row["asset_id"])
            if key in known_asset_keys:
                continue
            known_asset_keys.add(key)
            appended.append(row)
            page_appended += 1

        if page_appended == 0:
            stopped_on_known_page = True
            break
        if len(releases) < page["per_page"]:
            break
        if max_pages is not None and page_number >= max_pages:
            hit_page_limit = True
            break

        page_number += 1

    return (
        RepositoryMetadataSyncResult(
            repository=repository,
            pages_fetched=pages_fetched,
            assets_seen=assets_seen,
            assets_appended=len(appended),
            stopped_on_known_page=stopped_on_known_page,
            hit_page_limit=hit_page_limit,
        ),
        appended,
    )


async def run_metadata(
    *,
    grouped_specs: Mapping[Path, Iterable[str]],
    base_url: str,
    auth_token: str | None,
    per_page: int = MAX_RELEASES_PER_PAGE,
    max_pages: int | None = None,
) -> MetadataSyncResult:
    """Incrementally append newest adsb.lol GitHub release asset metadata.

    It treats the release history as an append-only newest-first log and stops
    per repository once a fetched page contains no unknown asset ids. It does
    not repair or backfill, in those cases delete the metadata JSONL and
    rerun to rebuild.
    """
    import httpx

    headers = github_headers(auth_token)
    file_results: list[MetadataFileSyncResult] = []

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=60.0,
        follow_redirects=True,
    ) as client:
        for output_path, repositories in grouped_specs.items():
            output_path = output_path.expanduser()
            known_asset_keys = _known_asset_keys(output_path)
            existing_assets = len(known_asset_keys)
            repository_results: list[RepositoryMetadataSyncResult] = []
            rows_to_append: list[AdsblolAsset] = []

            for repository in repositories:
                result, rows = await _sync_repository_incremental(
                    client,
                    repository,
                    known_asset_keys,
                    per_page=per_page,
                    max_pages=max_pages,
                )
                repository_results.append(result)
                rows_to_append.extend(rows)

            appended_assets = append_jsonl_rows_sync(
                output_path,
                rows_to_append,
            )
            file_results.append(
                MetadataFileSyncResult(
                    path=output_path,
                    existing_assets=existing_assets,
                    appended_assets=appended_assets,
                    repositories=tuple(repository_results),
                )
            )

    return MetadataSyncResult(files=tuple(file_results))


def append_jsonl_rows_sync(path: Path, rows: Iterable[AdsblolAsset]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


#
# metadata: parsing file names and analysing things
#


@dataclass(frozen=True, slots=True)
class AssetTimePeriod:
    asset_name: str
    start_at: datetime
    end_at: datetime

    @property
    def day(self) -> date:
        return self.start_at.date()


ModeVariant = Literal["prod", "staging", "mlatonly", "test"]


@dataclass(frozen=True, slots=True)
class ModeAssetPeriod:
    asset_name: str
    day: date
    variant: ModeVariant
    replica: str
    part: str | None


@dataclass(frozen=True, slots=True)
class ModeDailySizes:
    prod: int = 0
    staging: int = 0
    mlatonly: int = 0
    test: int = 0

    @property
    def total(self) -> int:
        return self.prod + self.staging + self.mlatonly + self.test


_DATALINK_RANGE_RE = re.compile(
    r"^adsblol-adl-(?P<bearer>[a-z0-9]+)_"
    r"(?P<start>\d{8}-\d{6})_(?P<end>\d{8}-\d{6})"
    r"(?:\.jsonl)?$"
)
_DATALINK_DAY_RE = re.compile(
    r"^adsblol-adl-(?P<day>\d{4}-\d{2}-\d{2})(?:\.jsonl)?$"
)
_ADSB_DAY_RE = re.compile(
    r"^v(?P<day>\d{4}\.\d{2}\.\d{2})-.*\.tar(?:\.[a-z0-9]{2,3})?$"
)
_MODE_DAY_RE = re.compile(
    r"^v(?P<day>\d{4}\.\d{2}\.\d{2})-planes-readsb-"
    r"(?P<variant>prod|staging|mlatonly|test)-"
    r"(?P<replica>\d+(?:tmp)?)\.tar"
    r"(?:\.(?P<part>[a-z]{2}))?$"
)


def _parse_utc_datetime(value: str, fmt: str) -> datetime:
    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)


def parse_asset_time_period(asset_name: str) -> AssetTimePeriod:
    base_name = asset_name.removesuffix(".zst")

    if match := _DATALINK_RANGE_RE.match(base_name):
        return AssetTimePeriod(
            asset_name=asset_name,
            start_at=_parse_utc_datetime(match["start"], "%Y%m%d-%H%M%S"),
            end_at=_parse_utc_datetime(match["end"], "%Y%m%d-%H%M%S"),
        )

    if match := _DATALINK_DAY_RE.match(base_name):
        start_at = _parse_utc_datetime(match["day"], "%Y-%m-%d")
        return AssetTimePeriod(
            asset_name=asset_name,
            start_at=start_at,
            end_at=start_at + timedelta(days=1),
        )

    if match := _ADSB_DAY_RE.match(base_name):
        start_at = _parse_utc_datetime(
            match["day"].replace(".", "-"),
            "%Y-%m-%d",
        )
        return AssetTimePeriod(
            asset_name=asset_name,
            start_at=start_at,
            end_at=start_at + timedelta(days=1),
        )

    raise ValueError(f"unrecognized asset name: {asset_name}")


def parse_mode_asset_period(asset_name: str) -> ModeAssetPeriod:
    if not (match := _MODE_DAY_RE.match(asset_name)):
        raise ValueError(f"unrecognized modes asset name: {asset_name}")

    return ModeAssetPeriod(
        asset_name=asset_name,
        day=_parse_utc_datetime(
            match["day"].replace(".", "-"),
            "%Y-%m-%d",
        ).date(),
        variant=cast(ModeVariant, match["variant"]),
        replica=match["replica"],
        part=match["part"],
    )


def _iter_input_paths(paths: Iterable[Path], stem: str) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob(f"{stem}.*"))
        elif path.name.startswith(stem):
            yield path


#
# datalinks: download
#


KNOWN_UNAVAILABLE_DATALINK_URLS = {
    "https://github.com/adsblol/aircraft-data-links-2026/releases/"
    "download/adsblol-adl-2026-05-14/"
    "adsblol-adl-vdl2_20260514-165234_20260514-165745.jsonl",
}  # for some reason they deleted it


def dir_datalinks(base_dir: Path) -> Path:
    return base_dir / "adsblol_datalinks"


@dataclass(frozen=True, slots=True)
class DatalinkDownloadItem:
    repository: str
    release_tag: str
    asset_name: str
    asset_day: date
    asset_size: int
    browser_download_url: str
    output_path: Path
    already_downloaded: bool


@dataclass(frozen=True, slots=True)
class DatalinkPlan:
    metadata_path: Path
    output_root: Path
    items: tuple[DatalinkDownloadItem, ...]
    selected_by_repo: Mapping[str, int]
    existing_by_repo: Mapping[str, int]

    @property
    def selected(self) -> int:
        return len(self.items)

    @property
    def existing(self) -> int:
        return sum(self.existing_by_repo.values())

    @property
    def missing(self) -> int:
        return self.selected - self.existing

    @property
    def missing_items(self) -> tuple[DatalinkDownloadItem, ...]:
        return tuple(item for item in self.items if not item.already_downloaded)


@dataclass(frozen=True, slots=True)
class DatalinkDownloadResult:
    plan: DatalinkPlan
    existing: int
    downloaded: int
    unavailable: int
    output_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _DownloadCounts:
    existing: int
    downloaded: int
    unavailable: int
    output_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class DatalinkDownloadFailure:
    item: DatalinkDownloadItem
    exception: Exception


def _datalink_jobs(jobs: int | None = None) -> int:
    return max(1, jobs if jobs is not None else min(8, os.cpu_count() or 1))


def _asset_output_name(asset_name: str) -> str:
    name = asset_name.removesuffix(".zst")
    return name if name.endswith(".jsonl") else f"{name}.jsonl"


def plan_datalinks(
    *,
    metadata_path: Path,
    output_root: Path,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
) -> DatalinkPlan:
    metadata_path = metadata_path.expanduser()
    output_root = output_root.expanduser()
    start = (
        start_date.date() if isinstance(start_date, datetime) else start_date
    )
    end = end_date.date() if isinstance(end_date, datetime) else end_date
    existing_jsonl = {
        path.name
        for path in output_root.glob("*.jsonl")
        if path.is_file() and path.stat().st_size > 0
    }
    selected_by_repo: Counter[str] = Counter()
    existing_by_repo: Counter[str] = Counter()
    items: list[DatalinkDownloadItem] = []
    seen_asset_keys: set[AssetKey] = set()
    seen_output_names: set[str] = set()

    for row in iter_metadata_rows(metadata_path):
        repository = row["repository"]
        if not repository.startswith("adsblol/aircraft-data-links-"):
            continue
        if row["browser_download_url"] in KNOWN_UNAVAILABLE_DATALINK_URLS:
            continue

        key = AssetKey(repository, int(row["asset_id"]))
        if key in seen_asset_keys:
            continue
        seen_asset_keys.add(key)

        asset_day = parse_asset_time_period(row["asset_name"]).day
        if start is not None and asset_day < start:
            continue
        if end is not None and asset_day > end:
            continue

        output_name = _asset_output_name(row["asset_name"])
        if output_name in seen_output_names:
            raise ValueError(f"duplicate datalink output name: {output_name}")
        seen_output_names.add(output_name)

        already_downloaded = output_name in existing_jsonl
        selected_by_repo[repository] += 1
        if already_downloaded:
            existing_by_repo[repository] += 1

        items.append(
            DatalinkDownloadItem(
                repository=repository,
                release_tag=row["release_tag"],
                asset_name=row["asset_name"],
                asset_day=asset_day,
                asset_size=int(row["asset_size"]),
                browser_download_url=row["browser_download_url"],
                output_path=output_root / output_name,
                already_downloaded=already_downloaded,
            )
        )

    return DatalinkPlan(
        metadata_path=metadata_path,
        output_root=output_root,
        items=tuple(
            sorted(
                items,
                key=lambda item: (
                    item.asset_day,
                    item.repository,
                    item.release_tag,
                    item.asset_name,
                ),
            )
        ),
        selected_by_repo=dict(sorted(selected_by_repo.items())),
        existing_by_repo=dict(sorted(existing_by_repo.items())),
    )


def _curl_args(url: str) -> list[str]:
    return [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--connect-timeout",
        "30",
        url,
    ]


def _commit_tmp(tmp: Path, dst: Path) -> None:
    if not tmp.is_file() or tmp.stat().st_size == 0:
        raise RuntimeError(f"empty output: {tmp}")
    tmp.replace(dst)


def _curl_to_jsonl(url: str, dst: Path, *, compressed: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f".{dst.name}.",
        suffix=".tmp",
        dir=dst.parent,
        delete=False,
    )
    tmp = Path(tmp_file.name)
    tmp_file.close()
    tmp.unlink(missing_ok=True)

    try:
        if not compressed:
            subprocess.run([*_curl_args(url), "--output", str(tmp)], check=True)
            _commit_tmp(tmp, dst)
            return

        with tmp.open("wb") as output_handle:
            curl = subprocess.Popen(_curl_args(url), stdout=subprocess.PIPE)
            zstd = subprocess.Popen(
                ["zstd", "-dcq", "-T1"],
                stdin=curl.stdout,
                stdout=output_handle,
            )
            if curl.stdout is not None:
                curl.stdout.close()
            zstd_return_code = zstd.wait()
            curl_return_code = curl.wait()

        if curl_return_code != 0:
            raise subprocess.CalledProcessError(curl_return_code, curl.args)
        if zstd_return_code != 0:
            raise subprocess.CalledProcessError(zstd_return_code, zstd.args)

        _commit_tmp(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _download_item(
    item: DatalinkDownloadItem,
) -> Literal["existing", "downloaded"]:
    if item.output_path.is_file() and item.output_path.stat().st_size > 0:
        return "existing"

    if item.asset_name.endswith(".zst"):
        _curl_to_jsonl(
            item.browser_download_url,
            item.output_path,
            compressed=True,
        )
        return "downloaded"

    try:
        _curl_to_jsonl(
            item.browser_download_url,
            item.output_path,
            compressed=False,
        )
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 22:
            raise
        _curl_to_jsonl(
            f"{item.browser_download_url}.zst",
            item.output_path,
            compressed=True,
        )
    return "downloaded"


def _download_missing_items(
    items: Sequence[DatalinkDownloadItem],
    *,
    jobs: int | None,
) -> _DownloadCounts:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    counts: Counter[str] = Counter()
    output_paths: list[Path] = []
    failures: list[DatalinkDownloadFailure] = []

    with ThreadPoolExecutor(max_workers=_datalink_jobs(jobs)) as executor:
        future_to_item = {
            executor.submit(_download_item, item): item for item in items
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                status = future.result()
            except Exception as exc:
                failures.append(DatalinkDownloadFailure(item, exc))
                continue
            counts[status] += 1
            if status in {"downloaded", "existing"}:
                output_paths.append(item.output_path)

    if failures:
        messages = "; ".join(
            f"{failure.item.output_path.name}: {failure.exception}"
            for failure in failures[:5]
        )
        raise RuntimeError(
            f"failed datalink downloads: {len(failures)}/{len(items)}; "
            f"{messages}"
        )

    return _DownloadCounts(
        existing=counts["existing"],
        downloaded=counts["downloaded"],
        unavailable=counts["unavailable"],
        output_paths=tuple(sorted(output_paths)),
    )


def datalinks_download(
    *,
    metadata_path: Path,
    output_root: Path,
    start_date: date | datetime | None = None,
    end_date: date | datetime | None = None,
    dry_run: bool = False,
    jobs: int | None = None,
) -> DatalinkDownloadResult:
    plan = plan_datalinks(
        metadata_path=metadata_path,
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
    )
    missing_items = plan.missing_items
    if dry_run or not missing_items:
        return DatalinkDownloadResult(
            plan=plan,
            existing=plan.existing,
            downloaded=0,
            unavailable=0,
            output_paths=tuple(
                item.output_path
                for item in plan.items
                if item.already_downloaded
            ),
        )

    result = _download_missing_items(missing_items, jobs=jobs)
    return DatalinkDownloadResult(
        plan=plan,
        existing=result.existing,
        downloaded=result.downloaded,
        unavailable=result.unavailable,
        output_paths=result.output_paths,
    )


#
# plotting
#


def load_metadata_daily_sizes(
    paths: Iterable[Path],
    stem: str,
) -> dict[date, int]:
    daily_sizes: dict[date, int] = {}
    seen_keys: set[AssetKey] = set()

    for path in _iter_input_paths(paths, stem):
        for row in iter_metadata_rows(path):
            key = AssetKey(row["repository"], int(row["asset_id"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            day = parse_asset_time_period(row["asset_name"]).day
            daily_sizes[day] = daily_sizes.get(day, 0) + int(row["asset_size"])

    return dict(sorted(daily_sizes.items()))


def load_modes_daily_sizes(
    paths: Iterable[Path],
    stem: str,
) -> dict[date, ModeDailySizes]:
    daily_sizes: dict[date, ModeDailySizes] = {}
    seen_keys: set[AssetKey] = set()

    for path in _iter_input_paths(paths, stem):
        for row in iter_metadata_rows(path):
            key = AssetKey(row["repository"], int(row["asset_id"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                period = parse_mode_asset_period(row["asset_name"])
            except ValueError:
                continue
            day_sizes = daily_sizes.setdefault(period.day, ModeDailySizes())
            asset_size = int(row["asset_size"])
            if period.variant == "prod":
                day_sizes = ModeDailySizes(
                    prod=day_sizes.prod + asset_size,
                    staging=day_sizes.staging,
                    mlatonly=day_sizes.mlatonly,
                    test=day_sizes.test,
                )
            elif period.variant == "staging":
                day_sizes = ModeDailySizes(
                    prod=day_sizes.prod,
                    staging=day_sizes.staging + asset_size,
                    mlatonly=day_sizes.mlatonly,
                    test=day_sizes.test,
                )
            elif period.variant == "mlatonly":
                day_sizes = ModeDailySizes(
                    prod=day_sizes.prod,
                    staging=day_sizes.staging,
                    mlatonly=day_sizes.mlatonly + asset_size,
                    test=day_sizes.test,
                )
            elif period.variant == "test":
                day_sizes = ModeDailySizes(
                    prod=day_sizes.prod,
                    staging=day_sizes.staging,
                    mlatonly=day_sizes.mlatonly,
                    test=day_sizes.test + asset_size,
                )
            daily_sizes[period.day] = day_sizes

    return dict(sorted(daily_sizes.items()))


def build_adsblol_figure(
    modes_daily_sizes: Mapping[date, ModeDailySizes],
    datalinks_daily_sizes: Mapping[date, int],
    *,
    line_color: str = "#5470c6",
    datalink_color: str = "#d9a15b",
    prod_color: str = "#5a9a8b",
    staging_color: str = "#c97c76",
    mlatonly_color: str = "#7d8a8c",
    test_color: str = "#8c7bd6",
) -> Figure:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    modes_days: list[Any] = list(modes_daily_sizes.keys())
    modes_prod_values = [modes_daily_sizes[day].prod for day in modes_days]
    modes_staging_values = [
        modes_daily_sizes[day].staging for day in modes_days
    ]
    modes_mlatonly_values = [
        modes_daily_sizes[day].mlatonly for day in modes_days
    ]
    modes_test_values = [modes_daily_sizes[day].test for day in modes_days]
    modes_total_values = [modes_daily_sizes[day].total for day in modes_days]
    modes_cumulative_values = list(accumulate(modes_total_values))

    dl_days: list[Any] = list(datalinks_daily_sizes.keys())
    dl_values = [datalinks_daily_sizes[day] for day in dl_days]
    dl_cumu_values = list(accumulate(dl_values))

    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0)
    ax_modes_top = fig.add_subplot(grid[0, 0])
    ax_modes_bottom = ax_modes_top.twinx()
    ax_dl_top = fig.add_subplot(grid[1, 0], sharex=ax_modes_top)
    ax_dl_bottom = ax_dl_top.twinx()

    ax_modes_top.plot(
        modes_days, modes_cumulative_values, color=prod_color, linewidth=2.4
    )
    ax_modes_bottom.bar(
        modes_days, modes_prod_values, color=prod_color, alpha=0.8, width=0.8
    )
    ax_modes_bottom.bar(
        modes_days,
        modes_staging_values,
        bottom=modes_prod_values,
        color=staging_color,
        alpha=0.55,
        width=0.8,
    )
    staging_bottom = [
        prod + staging
        for prod, staging in zip(modes_prod_values, modes_staging_values)
    ]
    ax_modes_bottom.bar(
        modes_days,
        modes_mlatonly_values,
        bottom=staging_bottom,
        color=mlatonly_color,
        alpha=0.5,
        width=0.8,
    )
    test_bottom = [
        bottom + mlatonly
        for bottom, mlatonly in zip(staging_bottom, modes_mlatonly_values)
    ]
    ax_modes_bottom.bar(
        modes_days,
        modes_test_values,
        bottom=test_bottom,
        color=test_color,
        alpha=0.45,
        width=0.8,
    )

    ax_dl_top.plot(dl_days, dl_cumu_values, color=line_color, linewidth=2.4)
    ax_dl_bottom.bar(
        dl_days, dl_values, color=datalink_color, alpha=0.28, width=0.8
    )

    ax_modes_top.set_ylabel("adsb cumulative raw hosted bytes")
    ax_modes_bottom.set_ylabel("daily raw hosted bytes")
    ax_modes_top.tick_params(axis="y", colors=prod_color)
    ax_modes_bottom.tick_params(axis="y")
    ax_modes_top.spines["left"].set_color(prod_color)
    ax_modes_top.set_ylim(bottom=0)
    ax_modes_bottom.set_ylim(bottom=0)
    ax_modes_top.tick_params(axis="x", labelbottom=False)

    ax_dl_top.set_ylabel(
        "datalinks cumulative raw hosted bytes", color=line_color
    )
    ax_dl_bottom.set_ylabel("daily raw hosted bytes", color=datalink_color)
    ax_dl_top.tick_params(axis="y", colors=line_color)
    ax_dl_bottom.tick_params(axis="y", colors=datalink_color)
    ax_dl_top.spines["left"].set_color(line_color)
    ax_dl_bottom.spines["right"].set_color(datalink_color)
    ax_dl_top.set_ylim(bottom=0)
    ax_dl_bottom.set_ylim(bottom=0)
    ax_dl_top.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
    ax_dl_bottom.tick_params(axis="x", bottom=False, labelbottom=False)

    legend_items = [
        ("prod", prod_color, 0.01),
        ("staging", staging_color, 0.07),
        ("mlatonly", mlatonly_color, 0.18),
        ("test", test_color, 0.30),
    ]
    for text, color, x in legend_items:
        ax_modes_top.text(
            x,
            0.98,
            text,
            color=color,
            transform=ax_modes_top.transAxes,
            va="top",
        )
    ax_dl_top.text(
        0.01,
        0.98,
        "datalinks",
        color=datalink_color,
        transform=ax_dl_top.transAxes,
        va="top",
    )

    ax_dl_top.xaxis.set_major_locator(
        mdates.AutoDateLocator()  # type: ignore[no-untyped-call]
    )
    ax_dl_top.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")  # type: ignore[no-untyped-call]
    )
    for label in ax_dl_top.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")  # type: ignore[attr-defined]

    fig.subplots_adjust(
        hspace=0,
        left=0.05,
        right=0.985,
        top=0.97,
        bottom=0.095,
    )
    return fig


#
# datalinks: JSONL row types
#

DatalinkProtocol = Literal["acars", "vdl2", "hfdl"]


class DatalinkAppRequired(TypedDict):
    name: str
    """Decoder name, e.g. 'acarsdec', 'dumpvdl2', or 'dumphfdl'"""

    ver: str
    """Decoder version/git revision, e.g. '8fcf327' or '2.4.0-dirty'"""


class DatalinkApp(DatalinkAppRequired, total=False):
    """See: https://github.com/sdr-enthusiasts/acars_router"""

    proxied: bool
    proxied_by: str  # acars_router
    acars_router_version: str  # 1.3.1
    acars_router_uuid: str  # 0ce9e2a9-...


class DatalinkAdsblolMeta(TypedDict):
    received_at: str
    """ISO-8601 UTC, e.g. '2025-11-22T00:18:44Z' or UTC offset"""
    protocol: DatalinkProtocol
    made_by: str  # e.g. github.com/iakat/adl-backend@...


class DatalinkTime(TypedDict):
    sec: int
    """Unix epoch seconds, UTC"""
    usec: int
    """Microsecond component"""


# flat acarsdec rows

AcarsFlightId = str
"""Flight id/number from ACARS, e.g. 'UA0097'. Not exactly the callsign"""
Registration = str
"""Aircraft registration, e.g. 'B-320S' or 'N876UA'"""


class AcarsDatalinkRequired(TypedDict):
    freq: t.FrequencyMhz[float]
    """VHF frequency in MHz, e.g. 131.45"""
    channel: int
    """Decoder input channel index, not one-to-one with freq; typically 0..11"""
    error: int
    """Decoder error/correction count; typically 0..3"""
    level: float
    """Decoder signal level metric; typically about -62.0..2.3"""
    timestamp: t.TimestampUtcS[float]
    """Decoder Unix epoch seconds, UTC; usually within ~1s of received_at"""
    app: DatalinkApp
    station_id: str
    """Receiver station id, e.g. 'RK-YBAF-ACARS'"""
    assstat: (
        Literal[
            "complete",
            "duplicate",
            "in progress",
            "out of sequence",
            "skipped",
        ]
        | str
    )
    mode: str
    """ACARS mode character; usually '2'"""
    label: str
    """ACARS message label, e.g. 'H1', 'Q0', 'MA'"""
    _adsblol: DatalinkAdsblolMeta


class AcarsDatalink(AcarsDatalinkRequired, total=False):
    block_id: str
    """ACARS block id; usually one digit, sometimes one letter"""
    ack: Literal[False] | str
    """False or one-character decoded ACK/NAK/control field"""
    tail: Registration
    text: str
    """Application text; may be empty or thousands of characters"""
    msgno: str
    """Message sequence number, commonly four chars, e.g. 'S35A'"""
    flight: AcarsFlightId


# embedded ACARS over VDL2/HFDL


class EmbeddedAcarsRequired(TypedDict):
    err: bool
    """Decoder error flag"""
    crc_ok: bool
    """Decoded ACARS CRC status"""
    more: bool
    """True when more message blocks follow"""
    reg: str
    """Aircraft registration, often with leading '.', e.g. '.VH-XZD'"""
    mode: str
    """ACARS mode character; typically '2'"""
    label: str
    """ACARS message label, e.g. 'H1', 'Q0', 'SA'"""
    blk_id: str
    ack: str
    """One-character decoded ACK/NAK/control field, e.g. '!'"""
    msg_text: str
    """ACARS application text; may be empty"""


class EmbeddedAcars(EmbeddedAcarsRequired, total=False):
    flight: AcarsFlightId
    msg_num: str
    """Message number, e.g. 'M23'"""
    msg_num_seq: str
    """Message number sequence component, e.g. 'A'"""
    sublabel: str
    """ACARS sublabel when decoded"""
    mfi: str
    """Message function identifier when decoded"""
    arinc622: dict[str, Any]
    # can also contain JSON key "media-adv". TODO switch to functional


# vdl2 / dumpvdl2 rows


class Vdl2AddressRequired(TypedDict):
    addr: str
    """VDL2 AVLC address, e.g. '7C77F7'"""  # is this icao hex?
    type: Literal["Aircraft", "Ground station"] | str


class Vdl2Address(Vdl2AddressRequired, total=False):
    status: Literal["Airborne", "On ground"] | str


class Vdl2Param(TypedDict):
    name: str
    """VDL2/XID parameter name"""
    value: Any
    """VDL2/XID parameter value; heterogeneous scalar/list/dict"""


class Vdl2XidRequired(TypedDict):
    err: bool
    """Decoder error flag; typically False"""
    type: (
        Literal[
            "GSIF",
            "XID_CMD_HO",
            "XID_CMD_LCR",
            "XID_CMD_LE",
            "XID_RSP_HO",
            "XID_RSP_LCR",
            "XID_RSP_LE",
        ]
        | str
    )
    """XID subtype"""
    type_descr: str
    """Human-readable XID subtype description"""
    vdl_params: list[Vdl2Param]
    """VDL-specific XID params; heterogeneous values"""


class Vdl2Xid(Vdl2XidRequired, total=False):
    pub_params: list[Vdl2Param]
    """Public XID params; heterogeneous values"""


class Vdl2AvlcRequired(TypedDict):
    cr: Literal["Command", "Response"] | str
    """AVLC command/response bit decoded as text"""
    dst: Vdl2Address
    frame_type: Literal["I", "U"] | str
    """AVLC frame type; typically information and unnumbered frames"""
    src: Vdl2Address


class Vdl2Avlc(Vdl2AvlcRequired, total=False):
    rseq: int
    """I-frame receive sequence number"""
    sseq: int
    """I-frame send sequence number"""
    poll: bool
    """I-frame poll bit"""
    cmd: Literal["DISC", "DM", "FRMR", "UA", "XID"] | str
    """U-frame command"""
    pf: bool
    """U-frame poll/final bit"""
    xid: Vdl2Xid
    acars: EmbeddedAcars


class Vdl2Payload(TypedDict):
    app: DatalinkApp
    avlc: Vdl2Avlc
    burst_len_octets: int
    """VDL2 burst length in octets; typically 13..1056"""
    freq: t.FrequencyHz[int]
    """RF frequency in Hz, e.g. 136975000"""
    idx: int
    """Decoder input/channel index; typically 0..3"""
    freq_skew: float  # TODO: units?
    """Decoder-estimated frequency skew/offset"""
    hdr_bits_fixed: int
    """Header bits corrected by decoder; typically 0..1"""
    noise_level: float  # TODO: units?
    octets_corrected_by_fec: int
    """Octets corrected by forward error correction; typically 0..8"""
    sig_level: float
    """Decoder signal level metric"""
    station: str
    """Receiver station id, e.g. 'RK-YBAF-VDL2'"""
    t: DatalinkTime


class Vdl2Datalink(TypedDict):
    vdl2: Vdl2Payload
    _adsblol: DatalinkAdsblolMeta


# hfdl / dumphfdl rows


class HfdlCodeDescription(TypedDict):
    code: int
    """Protocol/decoder numeric code"""
    descr: str
    """Human-readable code description"""


class HfdlTypeCode(TypedDict):
    name: str
    """Decoded type name, e.g. 'Frequency data' or 'Logon resume'"""
    id: int
    """Protocol numeric type id"""


class HfdlAircraftInfo(TypedDict, total=False):
    icao: str
    """ICAO hex address"""  # casing?
    manuf: str
    """Aircraft manufacturer"""
    model: str
    """Aircraft model"""
    opercode: str
    owner: str
    regnr: Registration
    typecode: str  # what does it look like?
    """Aircraft type"""


class HfdlEndpointRequired(TypedDict):
    type: Literal["Aircraft", "Ground station"] | str
    id: int
    """HFDL endpoint id; aircraft ids are dynamic within HFDL"""


class HfdlEndpoint(HfdlEndpointRequired, total=False):
    name: str
    """Ground station name when known, e.g. 'Shannon, Ireland'"""
    ac_info: HfdlAircraftInfo


class HfdlFrequency(TypedDict):
    id: int
    """Frequency table id"""
    freq: float  # kHz? 8843.0


class HfdlPosition(TypedDict):
    lat: t.LatitudeDeg[float]
    lon: t.LongitudeDeg[float]


class HfdlClockTime(TypedDict):
    """UTC"""

    hour: int
    min: int
    sec: int


class HfdlFrequencySearchCount(TypedDict):
    cur_leg: int
    prev_leg: int


class HfdlDisabledDuration(TypedDict):
    this_leg: int
    prev_leg: int


BitRateKey = str


class HfdlPduStats(TypedDict):
    mpdus_rx_ok_cnt: dict[BitRateKey, int]  # 300bps: 61
    mpdus_rx_err_cnt: dict[BitRateKey, int]
    mpdus_tx_cnt: dict[BitRateKey, int]
    mpdus_delivered_cnt: dict[BitRateKey, int]
    spdus_rx_ok_cnt: int
    spdus_missed_cnt: int


class HfdlFrequencyData(TypedDict):
    gs: HfdlEndpoint
    heard_on_freqs: list[HfdlFrequency]
    listening_on_freqs: list[HfdlFrequency]


class HfdlHfnpduRequired(TypedDict):
    err: bool
    type: HfdlTypeCode
    """HFNPDU type, e.g. Frequency data, Enveloped data, Performance data"""


class HfdlHfnpdu(HfdlHfnpduRequired, total=False):
    flight_id: str
    pos: HfdlPosition  # may contain sentinel-like values?
    utc_time: HfdlClockTime
    freq_data: list[HfdlFrequencyData]
    acars: EmbeddedAcars
    version: int
    time: HfdlClockTime
    flight_leg_num: int
    gs: HfdlEndpoint
    frequency: HfdlFrequency
    freq_search_cnt: HfdlFrequencySearchCount
    hfdl_disabled_duration: HfdlDisabledDuration
    pdu_stats: HfdlPduStats
    last_freq_change_cause: HfdlCodeDescription


class HfdlLpduRequired(TypedDict):
    err: bool
    dst: HfdlEndpoint
    src: HfdlEndpoint
    type: HfdlTypeCode


class HfdlLpdu(HfdlLpduRequired, total=False):
    hfnpdu: HfdlHfnpdu
    ac_info: HfdlAircraftInfo
    assigned_ac_id: int
    """Aircraft id assigned by ground station during logon confirm"""
    reason: HfdlCodeDescription


class HfdlGsStatus(TypedDict):
    gs: HfdlEndpoint
    utc_sync: bool
    """Whether ground station is UTC-synchronized"""
    freqs: list[HfdlFrequency]


class HfdlSpdu(TypedDict):
    err: bool
    src: HfdlEndpoint
    spdu_version: int
    rls: int
    iso: int
    change_note: int
    frame_index: int
    frame_offset: int
    min_priority: int
    systable_version: int
    gs_status: list[HfdlGsStatus]


class HfdlPayloadRequired(TypedDict):
    app: DatalinkApp
    freq: t.FrequencyHz[int]
    """RF frequency, e.g. 8942000"""
    noise_level: float  # TODO: units?
    sig_level: float  # TODO: units?
    station: str
    """Receiver station id, e.g. 'SS-EGBE-HFDL1'"""
    t: DatalinkTime
    bit_rate: t.BitPS[int]
    """HFDL bitrate; typically 300, 600, 1200"""
    freq_skew: float
    slot: Literal["S", "D"] | str


class HfdlPayload(HfdlPayloadRequired, total=False):
    lpdu: HfdlLpdu
    spdu: HfdlSpdu


class HfdlDatalink(TypedDict):
    hfdl: HfdlPayload
    _adsblol: DatalinkAdsblolMeta


Datalink = AcarsDatalink | Vdl2Datalink | HfdlDatalink
