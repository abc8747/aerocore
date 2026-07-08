"""Miscellaneous utility functions."""
# avoid importing anything other than stdlib, use local imports instead.

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Concatenate, Generic, ParamSpec, TypeVar

logger = getLogger(__name__)


def default_cache_dir() -> Path:
    """Requires optional dependency `platformdirs`"""
    import platformdirs

    return platformdirs.user_cache_path("aerocore") / "data"


def tree_map(function: Callable[[Any], Any], tree: Any) -> Any:
    """Apply `function` to every leaf in a small Python tree.

    This is similar to `jax.tree.map`, but does not support custom leaf
    predicates, cyclic structures, sets, named tuples, container subclasses,
    dataclass fields with `init=False`.

    The behaviour may change in the future.
    """  # keeping it super simple for now.
    if is_dataclass(tree) and not isinstance(tree, type):
        return replace(
            tree,
            **{
                field.name: tree_map(function, getattr(tree, field.name))
                for field in fields(tree)
            },
        )
    if type(tree) is tuple:
        return tuple(tree_map(function, value) for value in tree)
    if type(tree) is list:
        return [tree_map(function, value) for value in tree]
    if type(tree) is dict:
        return {key: tree_map(function, value) for key, value in tree.items()}
    return function(tree)


#
# hooks
#


P = ParamSpec("P")
S = TypeVar("S")


def hook(func: Callable[P, S]) -> Hook[P, S]:
    """Wraps a function in a `Hook` object, making it interceptable.

    Decorating a function with `@hook` allows its behavior to be observed,
    extended, or even completely replaced by downstream code.

    Example usage:
    ```py
    @hook
    def foo(state, t):
        ... # pure jax/torch/numpy code...
    ```
    For example, to time the function:
    ```py
    import time

    def timing_interceptor(original_fn, *args, **kwargs):
        start = time.perf_counter()
        result = original_fn(*args, **kwargs)
        end = time.perf_counter()
        print(f"{original_fn.__name__} took {end - start:.4f}s")
        return result

    foo.intercept(timing_interceptor)
    ```
    """
    return Hook(handler=func)


@dataclass(slots=True)
class Hook(Generic[P, S]):
    """A callable that implements the middleware pattern."""

    handler: Callable[P, S]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> S:
        return self.handler(*args, **kwargs)

    def intercept(
        self, interceptor: Callable[Concatenate[Callable[P, S], P], S]
    ) -> Hook[P, S]:
        original_handler = self.handler
        self.handler = lambda *args, **kwargs: interceptor(
            original_handler, *args, **kwargs
        )
        return self

    def debug(self) -> Hook[P, S]:
        """Log the function arguments and result."""

        def debug_interceptor(
            original_fn: Callable[P, S], /, *args: P.args, **kwargs: P.kwargs
        ) -> S:
            result = original_fn(*args, **kwargs)
            all_args = ", ".join(
                list(map(repr, args))
                + [f"{k}={v!r}" for k, v in kwargs.items()]
            )
            logger.debug("%s(%s) -> %r", original_fn.__name__, all_args, result)
            return result

        return self.intercept(debug_interceptor)
