from dataclasses import dataclass

from aerocore import hook
from aerocore.utils import tree_map


def test_hook() -> None:
    @hook
    def sub(a, b):  # type: ignore
        return a - b

    params = []

    @sub.intercept
    def sub_(original_fn, *args, **kwargs):  # type: ignore
        params.append((args, kwargs))
        return original_fn(*args, **kwargs)

    args = (3.1, 1.3)
    _ = sub(*args)
    assert (args, {}) in params


@dataclass(frozen=True)
class ExampleTree:
    value: int
    items: tuple[int, ...]
    nested: list[dict[str, int]]


def test_tree_map() -> None:
    tree = ExampleTree(
        value=1,
        items=(2, 3),
        nested=[{"unchanged-key": 4}],
    )
    assert tree_map(lambda value: value + 1, tree) == ExampleTree(
        value=2,
        items=(3, 4),
        nested=[{"unchanged-key": 5}],
    )
