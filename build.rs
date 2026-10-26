//! Build script.
//!
//! `cargo test` builds a test binary that *embeds* a Python interpreter (the extension
//! module is only linked against libpython when the `extension-module` feature is off).
//! On macOS the dynamic loader then needs an rpath to find `libpython` /
//! `Python3.framework`, otherwise the test binary aborts with
//! "Library not loaded: @rpath/Python3.framework/...". We add the interpreter's library
//! directory (and, for framework builds, the directory holding the `.framework`) to the
//! rpath. This is only done when the `extension-module` feature is off, i.e. never for
//! the wheels built by maturin (which enables that feature), so they are not affected.
//!
//! The interpreter used is the one pyo3 resolves: `PYO3_PYTHON` if set, otherwise the
//! first `python3` on `PATH`.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");

    if std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some() {
        return;
    }

    let config = pyo3_build_config::get();
    if let Some(lib_dir) = config.lib_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");

        // Framework build (e.g. /…/Python3.framework/Versions/3.x/lib): the install
        // name is "@rpath/Python3.framework/Versions/3.x/Python3", so the rpath must be
        // the directory that contains the .framework bundle.
        let mut dir = Path::new(lib_dir);
        while let Some(parent) = dir.parent() {
            if dir
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("framework"))
            {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", parent.display());
                break;
            }
            dir = parent;
        }
    }
}
