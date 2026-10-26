# CiTree (Rust core + Python API)

Python implementation, with a Rust-accelerated core, of the algorithms described in:

Ciampi, A., Lechevallier, Y., Castejón-Limas, M., González-Marcos, A. *Hierarchical clustering of subpopulations with a dissimilarity based on the likelihood ratio statistic: application to clustering massive data sets.* Pattern Anal Applic 11, 199–220 (2008). https://doi.org/10.1007/s10044-007-0088-4

The package is called `citree`. The Rust extension is exposed as `citree._rust`
(node classes `NormalNode`, `PoissonNode`, `MultinomialNode` and the functions
`cluster_normal_bins`, `cluster_poisson_bins`, `cluster_multinomial_bins`), and
`citree.hierarchical_clustering_rust` wraps it with the same bins/linkage interface
as the pure-Python `citree.hierarchical_clustering`.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages Python, the virtualenv and the build)
- A Rust toolchain (`rustup`), Rust ≥ 1.85
- Python ≥ 3.10 (uv will download one if needed; the project pins 3.12 in `.python-version`)

No system BLAS/LAPACK is needed: the small matrix inversions are done in pure Rust.

## Development

```bash
uv sync            # creates .venv, compiles the Rust extension with maturin, installs dev deps
uv run pytest      # Python tests (they also exercise the Rust extension)
cargo test         # Rust unit tests (embed a Python interpreter; no extension-module feature)
cargo clippy --all-targets
```

`uv sync` / `uv run` rebuild the extension automatically whenever `Cargo.toml`,
`src/**/*.rs` or `python/**/*.py` change (see `[tool.uv] cache-keys` in `pyproject.toml`).
For a faster edit/compile loop you can also run `uv run maturin develop` directly.

Release wheels:

```bash
uv run maturin build --release
```

## Usage

```python
import numpy as np
from citree import (
    stage_1_fast_clustering_kmeans,
    hierarchical_clustering_rust,
    cut_tree,
)

X = np.random.rand(5000, 3)
stage_1 = stage_1_fast_clustering_kmeans(data=X, number_of_bins=50, minimum_cardinality=10)
roots, linkage = hierarchical_clustering_rust(stage_1["leaf_bins"], strategy="normal")
labels = cut_tree(linkage, num_clusters=4, stage_1_cluster_labels=stage_1["clusters"])
```

## Layout

```
Cargo.toml            Rust crate (pyo3 0.30, rust-numpy 0.30, ndarray 0.17)
src/                  Rust core: lib.rs (generic clustering), normal/poisson/multinomial.rs,
                      linalg.rs (matrix inverse), pynodes.rs (Python bindings)
pyproject.toml        maturin build backend, dependencies, uv dependency groups
python/citree/        Python package (pure-Python reference implementation + Rust bridge)
tests/                pytest suite
```
