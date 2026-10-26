//! Python bindings (PyO3 + rust-numpy) for the CiTree node types and clustering functions.

use crate::{
    Node, NodeCount, NodeId, cluster_bins, multinomial::MultinomialNode as MultinomialNodeRust,
    normal::NormalNode as NormalNodeRust, poisson::PoissonNode as PoissonNodeRust,
};

use ndarray::{Array1, Array2};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::convert::Infallible;

// -------------------------- Array1Wrapper --------------------------

/// Owned 1-D `f64` array that converts to/from a NumPy array.
///
/// Extraction accepts any array-like (NumPy arrays of any numeric dtype, lists, tuples)
/// and casts it to `float64`.
#[derive(Debug, Clone)]
pub struct Array1Wrapper(pub Array1<f64>);

impl<'py> FromPyObject<'_, 'py> for Array1Wrapper {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let array: PyArrayLike1<'py, f64, AllowTypeChange> = ob.extract()?;
        Ok(Array1Wrapper(array.as_array().to_owned()))
    }
}

impl<'py> IntoPyObject<'py> for Array1Wrapper {
    type Target = PyArray1<f64>;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.into_pyarray(py))
    }
}

impl From<Array1Wrapper> for Array1<f64> {
    fn from(wrapper: Array1Wrapper) -> Self {
        wrapper.0
    }
}

// -------------------------- Array2Wrapper --------------------------

/// Owned 2-D `f64` array that converts to/from a NumPy array.
#[derive(Debug, Clone)]
pub struct Array2Wrapper(pub Array2<f64>);

impl<'py> FromPyObject<'_, 'py> for Array2Wrapper {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let array: PyArrayLike2<'py, f64, AllowTypeChange> = ob.extract()?;
        Ok(Array2Wrapper(array.as_array().to_owned()))
    }
}

impl<'py> IntoPyObject<'py> for Array2Wrapper {
    type Target = PyArray2<f64>;
    type Output = Bound<'py, Self::Target>;
    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.into_pyarray(py))
    }
}

impl From<Array2Wrapper> for Array2<f64> {
    fn from(wrapper: Array2Wrapper) -> Self {
        wrapper.0
    }
}

// -------------------------- Box<PyXxxNode> conversions --------------------------

// Children are stored as `Option<Box<Self>>`. `Box` is a fundamental type, so the orphan
// rules allow implementing PyO3's conversion traits for `Box<LocalType>` in this crate.
macro_rules! impl_boxed_node_conversions {
    ($node:ty) => {
        impl<'py> IntoPyObject<'py> for Box<$node> {
            type Target = $node;
            type Output = Bound<'py, Self::Target>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                Bound::new(py, *self)
            }
        }

        impl<'py> FromPyObject<'_, 'py> for Box<$node> {
            type Error = PyErr;

            fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
                let node = ob.cast::<$node>()?;
                Ok(Box::new(node.borrow().clone()))
            }
        }
    };
}

impl_boxed_node_conversions!(PyNormalNode);
impl_boxed_node_conversions!(PyPoissonNode);
impl_boxed_node_conversions!(PyMultinomialNode);

fn child_id<T>(child: &Option<Box<T>>, id: impl Fn(&T) -> usize) -> Option<usize> {
    child.as_deref().map(id)
}

// -------------------------- PyNormalNode --------------------------

#[pyclass(name = "NormalNode", module = "citree._rust", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyNormalNode {
    #[pyo3(get)]
    id: usize,

    #[pyo3(get)]
    height: f64,

    #[pyo3(get)]
    count: usize,

    #[pyo3(get)]
    left_child: Option<Box<PyNormalNode>>,

    #[pyo3(get)]
    right_child: Option<Box<PyNormalNode>>,

    #[pyo3(get)]
    x: Array1Wrapper,

    #[pyo3(get)]
    v: Array2Wrapper,

    #[pyo3(get)]
    inv_v: Array2Wrapper,
}

#[pymethods]
impl PyNormalNode {
    #[new]
    #[pyo3(signature = (id, height, count, x, v, inv_v, left_child=None, right_child=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        height: f64,
        count: usize,
        x: Array1Wrapper,
        v: Array2Wrapper,
        inv_v: Array2Wrapper,
        left_child: Option<Box<PyNormalNode>>,
        right_child: Option<Box<PyNormalNode>>,
    ) -> Self {
        PyNormalNode {
            id,
            height,
            count,
            left_child,
            right_child,
            x,
            v,
            inv_v,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "NormalNode(id={}, height={}, count={}, left_child_id={:?}, right_child_id={:?})",
            self.id,
            self.height,
            self.count,
            child_id(&self.left_child, |n| n.id),
            child_id(&self.right_child, |n| n.id),
        )
    }
}

impl From<NormalNodeRust> for PyNormalNode {
    fn from(node: NormalNodeRust) -> Self {
        let left_child = node
            .get_left_child()
            .map(|child| Box::new(PyNormalNode::from(child.clone())));
        let right_child = node
            .get_right_child()
            .map(|child| Box::new(PyNormalNode::from(child.clone())));

        PyNormalNode::new(
            *node.get_id(),
            node.get_height(),
            *node.get_count(),
            Array1Wrapper(node.get_x().clone()),
            Array2Wrapper(node.get_v().clone()),
            Array2Wrapper(node.get_inv_v().clone()),
            left_child,
            right_child,
        )
    }
}

impl From<PyNormalNode> for NormalNodeRust {
    fn from(py_node: PyNormalNode) -> Self {
        NormalNodeRust::new(
            NodeId(py_node.id),
            py_node.height,
            NodeCount(py_node.count),
            py_node
                .left_child
                .map(|child| Box::new(NormalNodeRust::from(*child))),
            py_node
                .right_child
                .map(|child| Box::new(NormalNodeRust::from(*child))),
            py_node.x.into(),
            py_node.v.into(),
            py_node.inv_v.into(),
        )
    }
}

// -------------------------- PyPoissonNode --------------------------

#[pyclass(name = "PoissonNode", module = "citree._rust", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPoissonNode {
    #[pyo3(get)]
    pub id: usize,

    #[pyo3(get)]
    pub height: f64,

    #[pyo3(get)]
    pub count: usize,

    #[pyo3(get)]
    pub left_child: Option<Box<PyPoissonNode>>,

    #[pyo3(get)]
    pub right_child: Option<Box<PyPoissonNode>>,

    #[pyo3(get)]
    pub n: f64,

    #[pyo3(get)]
    pub total: f64,

    #[pyo3(get)]
    pub ln_n_over_total: f64,
}

#[pymethods]
impl PyPoissonNode {
    #[new]
    #[pyo3(signature = (id, height, count, n, total, ln_n_over_total, left_child=None, right_child=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        height: f64,
        count: usize,
        n: f64,
        total: f64,
        ln_n_over_total: f64,
        left_child: Option<Box<PyPoissonNode>>,
        right_child: Option<Box<PyPoissonNode>>,
    ) -> Self {
        PyPoissonNode {
            id,
            height,
            count,
            left_child,
            right_child,
            n,
            total,
            ln_n_over_total,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "PoissonNode(id={}, height={}, count={}, left_child_id={:?}, right_child_id={:?}, n={}, total={}, ln_n_over_total={})",
            self.id,
            self.height,
            self.count,
            child_id(&self.left_child, |n| n.id),
            child_id(&self.right_child, |n| n.id),
            self.n,
            self.total,
            self.ln_n_over_total
        )
    }
}

impl From<PoissonNodeRust> for PyPoissonNode {
    fn from(node: PoissonNodeRust) -> Self {
        let left_child = node
            .get_left_child()
            .map(|child| Box::new(PyPoissonNode::from(child.clone())));
        let right_child = node
            .get_right_child()
            .map(|child| Box::new(PyPoissonNode::from(child.clone())));

        PyPoissonNode::new(
            *node.get_id(),
            node.get_height(),
            *node.get_count(),
            node.get_n(),
            node.get_total(),
            node.get_ln_n_over_total(),
            left_child,
            right_child,
        )
    }
}

impl From<PyPoissonNode> for PoissonNodeRust {
    fn from(py_node: PyPoissonNode) -> Self {
        PoissonNodeRust::new(
            NodeId(py_node.id),
            py_node.height,
            NodeCount(py_node.count),
            py_node
                .left_child
                .map(|child| Box::new(PoissonNodeRust::from(*child))),
            py_node
                .right_child
                .map(|child| Box::new(PoissonNodeRust::from(*child))),
            py_node.n,
            py_node.total,
            py_node.ln_n_over_total,
        )
    }
}

// -------------------------- PyMultinomialNode --------------------------

#[pyclass(name = "MultinomialNode", module = "citree._rust", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMultinomialNode {
    #[pyo3(get)]
    id: usize,

    #[pyo3(get)]
    height: f64,

    #[pyo3(get)]
    count: usize,

    #[pyo3(get)]
    left_child: Option<Box<PyMultinomialNode>>,

    #[pyo3(get)]
    right_child: Option<Box<PyMultinomialNode>>,

    #[pyo3(get)]
    n: Array1Wrapper,

    #[pyo3(get)]
    total: f64,

    #[pyo3(get)]
    ln_n_over_total: Array1Wrapper,
}

#[pymethods]
impl PyMultinomialNode {
    #[new]
    #[pyo3(signature = (id, height, count, n, total, ln_n_over_total, left_child=None, right_child=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: usize,
        height: f64,
        count: usize,
        n: Array1Wrapper,
        total: f64,
        ln_n_over_total: Array1Wrapper,
        left_child: Option<Box<PyMultinomialNode>>,
        right_child: Option<Box<PyMultinomialNode>>,
    ) -> Self {
        PyMultinomialNode {
            id,
            height,
            count,
            left_child,
            right_child,
            n,
            total,
            ln_n_over_total,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MultinomialNode(id={}, height={}, count={}, left_child_id={:?}, right_child_id={:?}, n={}, total={})",
            self.id,
            self.height,
            self.count,
            child_id(&self.left_child, |n| n.id),
            child_id(&self.right_child, |n| n.id),
            self.n.0,
            self.total
        )
    }
}

impl From<MultinomialNodeRust> for PyMultinomialNode {
    fn from(node: MultinomialNodeRust) -> Self {
        let left_child = node
            .get_left_child()
            .map(|child| Box::new(PyMultinomialNode::from(child.clone())));
        let right_child = node
            .get_right_child()
            .map(|child| Box::new(PyMultinomialNode::from(child.clone())));

        PyMultinomialNode::new(
            *node.get_id(),
            node.get_height(),
            *node.get_count(),
            Array1Wrapper(node.get_n().clone()),
            node.get_total(),
            Array1Wrapper(node.get_ln_n_over_total().clone()),
            left_child,
            right_child,
        )
    }
}

impl From<PyMultinomialNode> for MultinomialNodeRust {
    fn from(py_node: PyMultinomialNode) -> Self {
        MultinomialNodeRust::new(
            NodeId(py_node.id),
            py_node.height,
            NodeCount(py_node.count),
            py_node
                .left_child
                .map(|child| Box::new(MultinomialNodeRust::from(*child))),
            py_node
                .right_child
                .map(|child| Box::new(MultinomialNodeRust::from(*child))),
            py_node.n.into(),
            py_node.total,
            py_node.ln_n_over_total.into(),
        )
    }
}

// -------------------------- Python functions ----------------------------

/// Runs the agglomerative clustering on Rust nodes without holding the Python
/// thread state, and converts the result back to the Python node type.
fn cluster_py<R, P>(py: Python<'_>, bins: Vec<P>, kind: &str) -> PyResult<(Array2Wrapper, P)>
where
    R: Node + Clone + From<P> + Send + Sync,
    P: From<R>,
{
    let rust_bins: Vec<R> = bins.into_iter().map(R::from).collect();
    let (linkage, dendrogram) = py
        .detach(|| cluster_bins::<R>(&rust_bins))
        .map_err(|e| PyRuntimeError::new_err(format!("Error clustering {kind} nodes: {e:#}")))?;
    Ok((Array2Wrapper(linkage), P::from(dendrogram)))
}

/// Clusters Normal leaf nodes. Returns `(linkage_matrix, root_node)`.
#[pyfunction]
#[pyo3(name = "cluster_normal_bins")]
pub fn cluster_normal_bins_py(
    py: Python<'_>,
    bins: Vec<PyNormalNode>,
) -> PyResult<(Array2Wrapper, PyNormalNode)> {
    cluster_py::<NormalNodeRust, _>(py, bins, "normal")
}

/// Clusters Poisson leaf nodes. Returns `(linkage_matrix, root_node)`.
#[pyfunction]
#[pyo3(name = "cluster_poisson_bins")]
pub fn cluster_poisson_bins_py(
    py: Python<'_>,
    bins: Vec<PyPoissonNode>,
) -> PyResult<(Array2Wrapper, PyPoissonNode)> {
    cluster_py::<PoissonNodeRust, _>(py, bins, "poisson")
}

/// Clusters Multinomial leaf nodes. Returns `(linkage_matrix, root_node)`.
#[pyfunction]
#[pyo3(name = "cluster_multinomial_bins")]
pub fn cluster_multinomial_bins_py(
    py: Python<'_>,
    bins: Vec<PyMultinomialNode>,
) -> PyResult<(Array2Wrapper, PyMultinomialNode)> {
    cluster_py::<MultinomialNodeRust, _>(py, bins, "multinomial")
}

// -------------------------- PYTHON MODULE --------------------------

#[pymodule(name = "_rust")]
fn citree_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyNormalNode>()?;
    m.add_class::<PyPoissonNode>()?;
    m.add_class::<PyMultinomialNode>()?;
    // Backwards-compatible alias: the Poisson class used to be exported as `PyPoissonNode`.
    m.add("PyPoissonNode", m.getattr("PoissonNode")?)?;

    // Functions
    m.add_function(wrap_pyfunction!(cluster_normal_bins_py, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_poisson_bins_py, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_multinomial_bins_py, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
