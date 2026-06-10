"""GitHub release metadata for ADSBLOL globe-history snapshots.

Requires extras:

- `networking`
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, TypedDict

import httpx

DEFAULT_GH_API_BASE_URL = "https://api.github.com"
MAX_RELEASES_PER_PAGE = 100
logger = logging.getLogger(__name__)


class GitHubReleaseAsset(TypedDict):
    id: int
    node_id: str
    name: str
    label: str | None
    uploader: dict[str, Any] | None
    content_type: str
    state: str
    size: int
    digest: str | None
    download_count: int
    created_at: str
    updated_at: str
    browser_download_url: str
    url: str


class GitHubRelease(TypedDict):
    id: int
    tag_name: str
    name: str | None
    created_at: str
    published_at: str | None
    html_url: str
    assets: list[GitHubReleaseAsset]


class ReleasePage(TypedDict):
    repository: str
    page: int
    per_page: int
    releases: list[GitHubRelease]


class AssetSummary(TypedDict):
    repository: str
    release_id: int
    release_tag: str
    release_name: str | None
    release_created_at: str
    release_published_at: str | None
    release_html_url: str
    asset_id: int
    asset_name: str
    asset_size: int
    asset_content_type: str
    asset_digest: str | None
    asset_created_at: str
    asset_updated_at: str
    api_asset_url: str
    browser_download_url: str


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
        logger.warning("no github auth token found, you may be rate limited")
        return None

    token = result.stdout.strip()
    if not token:
        return None
    return token


def github_headers(auth_token: str | None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if auth_token is not None:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


async def fetch_release_page(
    client: httpx.AsyncClient,
    repository: str,
    *,
    page: int = 1,
    per_page: int = MAX_RELEASES_PER_PAGE,
) -> ReleasePage:
    response = await client.get(
        f"/repos/{repository}/releases",
        params={"page": page, "per_page": per_page},
    )
    response.raise_for_status()

    releases: list[GitHubRelease] = response.json()
    return {
        "repository": repository,
        "page": page,
        "per_page": per_page,
        "releases": releases,
    }


def iter_release_asset_downloads(page: ReleasePage) -> Iterator[AssetSummary]:
    for release in page["releases"]:
        for asset in release["assets"]:
            yield AssetSummary(
                repository=page["repository"],
                release_id=release["id"],
                release_tag=release["tag_name"],
                release_name=release["name"],
                release_created_at=release["created_at"],
                release_published_at=release["published_at"],
                release_html_url=release["html_url"],
                asset_id=asset["id"],
                asset_name=asset["name"],
                asset_size=asset["size"],
                asset_content_type=asset["content_type"],
                asset_digest=asset["digest"],
                asset_created_at=asset["created_at"],
                asset_updated_at=asset["updated_at"],
                api_asset_url=asset["url"],
                browser_download_url=asset["browser_download_url"],
            )


def load_known_asset_keys(path: Path) -> set[tuple[str, int]]:
    known_asset_keys: set[tuple[str, int]] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (stripped := line.strip()):
                continue
            payload: AssetSummary = json.loads(stripped)
            known_asset_keys.add((payload["repository"], payload["asset_id"]))
    return known_asset_keys


async def download_repository_pages(
    client: httpx.AsyncClient,
    repository: str,
    known_asset_keys: set[tuple[str, int]],
) -> AsyncIterator[AssetSummary]:
    page_number = 1

    while True:
        page = await fetch_release_page(
            client,
            repository=repository,
            page=page_number,
            per_page=MAX_RELEASES_PER_PAGE,
        )
        if not (releases := page["releases"]):
            break

        yielded_any = False
        for row in iter_release_asset_downloads(page):
            if (
                key := (row["repository"], row["asset_id"])
            ) in known_asset_keys:
                continue
            yielded_any = True
            known_asset_keys.add(key)
            yield row

        if not yielded_any:
            break

        if len(releases) < page["per_page"]:
            break

        page_number += 1


async def append_jsonl_rows(
    path: Path, rows: AsyncIterator[AssetSummary]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    appended_count = 0
    with path.open("a", encoding="utf-8") as handle:
        async for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            appended_count += 1
    return appended_count


async def metadata_async(
    *,
    grouped_specs: dict[Path, list[str]],
    base_url: str,
    auth_token: str | None,
) -> None:
    headers = github_headers(auth_token)
    total_appended_rows = 0
    total_existing_asset_count = 0

    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=30.0,
        follow_redirects=True,
    ) as client:
        for output_path, repositories in grouped_specs.items():
            known_asset_keys = (
                load_known_asset_keys(output_path)
                if output_path.exists()
                else set()
            )
            total_existing_asset_count += len(known_asset_keys)
            appended_count = 0
            logger.info(f"found {total_existing_asset_count=} at {output_path}")

            for repository in repositories:
                appended_count += await append_jsonl_rows(
                    output_path,
                    download_repository_pages(
                        client,
                        repository,
                        known_asset_keys,
                    ),
                )
                logger.info(f"wrote {appended_count=} rows to {output_path}")

            total_appended_rows += appended_count

    output_count = len(grouped_specs)
    logger.info(
        f"{output_count=} {total_existing_asset_count=} {total_appended_rows=}"
    )
