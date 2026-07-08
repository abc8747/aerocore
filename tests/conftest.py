from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def acropole_data_paths() -> tuple[Path, ...]:
    import asyncio

    from aerocore.acropole import sync_data

    return asyncio.run(sync_data())
