from ..core.registries import register

# canonical word-boundary symbol
register("boundary_scheme", "word")(lambda: "#")
