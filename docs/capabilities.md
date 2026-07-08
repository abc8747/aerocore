# Capabilities

Most aerocore functions can be used with multiple computational backends by passing an Array API-compatible namespace to the `xp` keyword argument.

This allows us to do some cool tricks like automatically differentiating and vmapping a function via JAX:

```python
--8<-- "examples/capabilities_jax.py:input0"
```

```text
--8<-- "examples/capabilities_jax.py:output0"
```

And even pipe an entire dataframe column through an aerocore function with Polars Lazy evaluation:

```python
--8<-- "examples/capabilities_polars.py:input0"
```

```text
--8<-- "examples/capabilities_polars.py:output0"
```

```python
--8<-- "examples/capabilities_polars.py:input1"
```

```text
--8<-- "examples/capabilities_polars.py:output1"
```
