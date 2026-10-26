"""Tests of the Rust-backed clustering exposed through ``citree._rust``."""

import numpy as np
import pandas as pd
import pytest

import citree
from citree import _rust as ct
from citree import (
    MultinomialBin,
    Node,
    NormalBin,
    PoissonBin,
    get_tree_ids_from,
    hierarchical_clustering,
    hierarchical_clustering_rust,
    stage_1_fast_clustering_kmeans,
    stage_2_strategies,
)


def test_module_exports():
    assert isinstance(citree.__version__, str)
    for name in (
        "NormalNode",
        "PoissonNode",
        "MultinomialNode",
        "cluster_normal_bins",
        "cluster_poisson_bins",
        "cluster_multinomial_bins",
    ):
        assert hasattr(ct, name), name


def test_normal_node_roundtrip():
    x = np.array([1.0, 2.0])
    v = np.array([[2.0, 0.5], [0.5, 1.0]])
    inv_v = np.linalg.inv(v)
    leaf = ct.NormalNode(id=3, height=0.0, count=7, x=x, v=v, inv_v=inv_v)
    node = ct.NormalNode(id=4, height=1.5, count=7, x=x, v=v, inv_v=inv_v, left_child=leaf)

    assert node.id == 4
    assert node.count == 7
    assert node.right_child is None
    assert node.left_child.id == 3
    np.testing.assert_allclose(node.x, x)
    np.testing.assert_allclose(node.left_child.inv_v, inv_v)
    assert "NormalNode(id=4" in repr(node)


def test_accepts_integer_arrays():
    # Integer NumPy input is cast to float64 instead of raising.
    node = ct.MultinomialNode(
        id=0, height=0.0, count=1, n=np.array([2, 1]), total=3.0, ln_n_over_total=np.log([2, 1])
    )
    assert node.n.dtype == np.float64


def test_cluster_poisson_bins_matches_python():
    bins = [
        PoissonBin(node=Node(id=i, height=0.0), n=n, N=N)
        for i, (n, N) in enumerate([(3, 5), (2, 10), (8, 9), (1, 7), (5, 6)])
    ]
    py_roots, py_linkage = hierarchical_clustering(list(bins), stage_2_strategies["poisson"])
    rs_roots, rs_linkage = hierarchical_clustering_rust(bins, strategy="poisson")

    assert rs_linkage.shape == py_linkage.shape == (4, 4)
    # Same merge order (columns 0 and 1) ...
    np.testing.assert_array_equal(rs_linkage[:, :2], py_linkage[:, :2])
    # ... and the same tree underneath the root.
    assert sorted(get_tree_ids_from(rs_roots[0])) == sorted(get_tree_ids_from(py_roots[0]))
    assert rs_roots[0].node.id == py_roots[0].node.id
    assert rs_roots[0].n == pytest.approx(sum(b.n for b in bins))


def test_cluster_multinomial_bins_matches_python():
    counts = [[2, 3, 1], [1, 2, 4], [3, 6, 4], [2, 5, 4], [9, 1, 1]]
    bins = [MultinomialBin(node=Node(id=i, height=0.0), n=np.array(c)) for i, c in enumerate(counts)]
    _, py_linkage = hierarchical_clustering(list(bins), stage_2_strategies["multinomial"])
    rs_roots, rs_linkage = hierarchical_clustering_rust(bins, strategy="multinomial")

    np.testing.assert_array_equal(rs_linkage[:, :2], py_linkage[:, :2])
    np.testing.assert_allclose(rs_roots[0].n, np.sum(counts, axis=0))


def test_cluster_normal_bins_matches_python():
    X = pd.read_csv("tests/media/datos.csv", index_col="indice").drop(["Bin", "Class"], axis=1)
    stage_1 = stage_1_fast_clustering_kmeans(data=X.values, number_of_bins=20, minimum_cardinality=10)
    bins = stage_1["leaf_bins"]

    py_roots, py_linkage = hierarchical_clustering(list(bins), stage_2_strategies["normal"])
    rs_roots, rs_linkage = hierarchical_clustering_rust(bins, strategy="normal")

    assert len(rs_roots) == 1
    assert rs_linkage.shape == (19, 4)
    np.testing.assert_array_equal(rs_linkage[:, :2], py_linkage[:, :2])
    assert set(get_tree_ids_from(rs_roots[0])) == set(range(20))
    # The root node of the Rust tree represents every observation of the leaves.
    assert rs_roots[0].node.count == sum(b.node.count for b in bins)
    assert isinstance(rs_roots[0], NormalBin)


def test_cluster_rejects_bad_ids():
    bins = [
        ct.PoissonNode(id=i, height=0.0, count=1, n=1.0, total=2.0, ln_n_over_total=np.log(0.5))
        for i in (0, 2)  # not consecutive
    ]
    with pytest.raises(RuntimeError, match="Non consecutive"):
        ct.cluster_poisson_bins(bins)
    with pytest.raises(RuntimeError, match="Empty"):
        ct.cluster_poisson_bins([])
