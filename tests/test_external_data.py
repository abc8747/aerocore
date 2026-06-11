import json
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import pytest
from httpx import AsyncClient

network = pytest.mark.network


@dataclass(frozen=True, slots=True)
class AdsblolMetadataFixture:
    path: Path
    row_count: int
    pages_fetched: int
    assets_appended: int


@pytest.fixture(scope="session")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(http2=True) as client:
        yield client


@pytest.fixture(scope="session")
async def adsblol_2025_metadata(
    tmp_path_factory: pytest.TempPathFactory,
) -> AdsblolMetadataFixture:
    from aerocore.data import adsblol

    path = tmp_path_factory.mktemp("adsblol") / "datalinks-2025.jsonl"
    result = await adsblol.run_metadata(
        grouped_specs={path: ["adsblol/aircraft-data-links-2025"]},
        base_url=adsblol.DEFAULT_GH_API_BASE_URL,
        auth_token=adsblol.get_github_auth_token(),
        per_page=5,
        max_pages=1,
    )
    rows = list(adsblol.iter_metadata_rows(path))
    assert rows
    repo_result = result.files[0].repositories[0]
    return AdsblolMetadataFixture(
        path=path,
        row_count=len(rows),
        pages_fetched=repo_result.pages_fetched,
        assets_appended=repo_result.assets_appended,
    )


@pytest.mark.anyio
@network
async def test_airports(async_client: AsyncClient) -> None:
    from aerocore.data.airports import fetch_airports

    airports = await fetch_airports(async_client)
    assert len(airports) > 60_000


@pytest.mark.anyio
@network
async def test_emissions_data(async_client: AsyncClient) -> None:
    from aerocore.data.engine_emissions import fetch_emissions_data

    emissions_data = await fetch_emissions_data(async_client)
    assert len(emissions_data.data) > 800


@pytest.mark.anyio
@network
async def test_aircraft_types(async_client: AsyncClient) -> None:
    from aerocore.data.aircraft_types import fetch_aircraft_types

    aircraft_types = await fetch_aircraft_types(async_client)
    assert len(aircraft_types) > 7_000


@pytest.mark.anyio
@network
async def test_manufacturers(async_client: AsyncClient) -> None:
    from aerocore.data.aircraft_types import fetch_manufacturers

    manufacturers = await fetch_manufacturers(async_client)
    assert len(manufacturers) > 2_000


#
# adsblol
#


@pytest.mark.anyio
@network
async def test_adsblol_metadata_sync_downloads_2025_first_page(
    adsblol_2025_metadata: AdsblolMetadataFixture,
) -> None:
    from aerocore.data import adsblol

    rows = list(adsblol.iter_metadata_rows(adsblol_2025_metadata.path))
    keys = {
        adsblol.AssetKey(row["repository"], row["asset_id"]) for row in rows
    }

    assert adsblol_2025_metadata.pages_fetched == 1
    assert adsblol_2025_metadata.assets_appended == len(rows)
    assert len(rows) == adsblol_2025_metadata.row_count
    assert len(keys) == len(rows)
    assert {row["repository"] for row in rows} == {
        "adsblol/aircraft-data-links-2025",
    }
    assert all(
        row["asset_digest"] is None or row["asset_digest"] for row in rows
    )
    assert all(row["browser_download_url"] for row in rows)


@pytest.mark.anyio
@network
async def test_adsblol_metadata_sync_stops_on_known_first_page(
    adsblol_2025_metadata: AdsblolMetadataFixture,
) -> None:
    from aerocore.data import adsblol

    before = list(adsblol.iter_metadata_rows(adsblol_2025_metadata.path))
    result = await adsblol.run_metadata(
        grouped_specs={
            adsblol_2025_metadata.path: [
                "adsblol/aircraft-data-links-2025",
            ],
        },
        base_url=adsblol.DEFAULT_GH_API_BASE_URL,
        auth_token=adsblol.get_github_auth_token(),
        per_page=5,
        max_pages=2,
    )
    after = list(adsblol.iter_metadata_rows(adsblol_2025_metadata.path))
    repo_result = result.files[0].repositories[0]

    assert repo_result.pages_fetched == 1
    assert repo_result.stopped_on_known_page
    assert repo_result.assets_appended == 0
    assert result.files[0].appended_assets == 0
    assert [row["asset_id"] for row in after] == [
        row["asset_id"] for row in before
    ]


@pytest.mark.anyio
@network
async def test_adsblol_datalinks_downloads_and_extracts_zst(
    tmp_path: Path,
    adsblol_2025_metadata: AdsblolMetadataFixture,
) -> None:
    from aerocore.data import adsblol

    rows = [
        row
        for row in adsblol.iter_metadata_rows(adsblol_2025_metadata.path)
        if row["asset_name"].endswith(".jsonl.zst")
        and row["browser_download_url"]
        not in adsblol.KNOWN_UNAVAILABLE_DATALINK_URLS
        and int(row["asset_size"]) > 1_000
    ]
    assert rows

    row = min(rows, key=lambda payload: int(payload["asset_size"]))
    metadata_path = tmp_path / "single-datalink.jsonl"
    metadata_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    period = adsblol.parse_asset_time_period(row["asset_name"])
    output_root = tmp_path / "datalinks"
    first_result = adsblol.datalinks_download(
        metadata_path=metadata_path,
        output_root=output_root,
        start_date=period.day,
        end_date=period.day,
        jobs=1,
    )

    output_path = output_root / row["asset_name"].removesuffix(".zst")
    assert first_result.downloaded == 1
    assert first_result.existing == 0
    assert first_result.output_paths == (output_path,)
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    first_line = output_path.read_text(encoding="utf-8").splitlines()[0]
    assert isinstance(json.loads(first_line), dict)
    assert not (output_root / row["asset_name"]).exists()

    second_result = adsblol.datalinks_download(
        metadata_path=metadata_path,
        output_root=output_root,
        start_date=period.day,
        end_date=period.day,
        jobs=1,
    )
    assert second_result.existing == 1
    assert second_result.downloaded == 0
