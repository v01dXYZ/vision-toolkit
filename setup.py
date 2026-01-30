import os
import pathlib
import numpy as np

from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

SRC_DIR = pathlib.Path("src")

def get_cython_pkgs_and_ext_modules():
    ext_modules = []
    pkgs = []

    for p in (SRC_DIR / "vision_toolkit").glob("**/*.pyx"):
        pkg_parts = p.with_suffix("").parts
        pkgs.append(".".join(pkg_parts[1:-1]))
        ext_modules.append(
            Extension(
                ".".join(pkg_parts[1:]),
                sources=[p],
            )
        )

    return pkgs, ext_modules

cython_pkgs, cython_ext_modules = get_cython_pkgs_and_ext_modules()

VISION_TOOLKIT_BUILD = os.getenv("VISION_TOOLKIT_BUILD", "all").lower()

kwargs = {
    "version": "0.1",
    "package_dir": {"": "src"},
}

c_kwargs = {
    "packages": cython_pkgs,
    "ext_modules": cythonize(cython_ext_modules, language_level="3"),
    "include_dirs": [np.get_include()],
}
py_kwargs = {
    "packages": find_packages(where="src"),
}

# The following is a hack to allow separately building and packaging
# the Cython package and the Python package.
# It is meant to reduce CI time.
if VISION_TOOLKIT_BUILD == "c":
    kwargs.update(c_kwargs)
    kwargs["name"] = "vision_toolkit_c"
else:
    if VISION_TOOLKIT_BUILD != "py":
        kwargs.update(c_kwargs)
    kwargs.update(py_kwargs)
    kwargs["name"] = "vision_toolkit"

setup(**kwargs)
