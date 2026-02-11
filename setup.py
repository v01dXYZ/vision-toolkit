import os
import pathlib
import numpy as np

from setuptools import Extension, setup, find_packages, find_namespace_packages
from Cython.Build import cythonize

VISION_TOOLKIT_BUILD = os.getenv("VISION_TOOLKIT_BUILD", "all").lower()

SEP_LIST = ("-", "/", ".")
COVERAGE_KWS = ("cov", "coverage")

IS_COVERAGE = VISION_TOOLKIT_BUILD in COVERAGE_KWS

if not IS_COVERAGE:
    for cov_kw in COVERAGE_KWS:
        for sep in SEP_LIST:
            suffix = f"{sep}{cov_kw}"
            prefix = f"{cov_kw}{sep}"

            if VISION_TOOLKIT_BUILD.startswith(prefix):
                IS_COVERAGE = True
                VISION_TOOLKIT_BUILD = VISION_TOOLKIT_BUILD.removeprefix(prefix)
                break

            if VISION_TOOLKIT_BUILD.endswith(suffix):
                IS_COVERAGE = True
                VISION_TOOLKIT_BUILD = VISION_TOOLKIT_BUILD.removesuffix(suffix)
                break

SRC_DIR = pathlib.Path("src")

def get_cython_pkgs_and_ext_modules():
    ext_modules = []
    pkgs = []


    define_macros = [
        ("CYTHON_TRACE_NOGIL", "1"),
        # ("CYTHON_TRACE", "1"), # the macro above implies this one as well
        # ---
        # As of 2026, coverage has a "sysmon" core (for sys.monitoring)
        # BUT Plugin filetracers are not supported with SysMonitoring.
        # Since Cython default to sys.monitoring with 3.13+,
        # we explicitly have to set it to use the old sys.settrace.
        ("CYTHON_USE_SYS_MONITORING", "0"),
    ] if IS_COVERAGE else []

    for p in SRC_DIR.glob("**/*.pyx"):
        pkg_parts = p.with_suffix("").parts
        pkgs.append(".".join(pkg_parts[1:-1]))
        ext_modules.append(
            Extension(
                ".".join(pkg_parts[1:]),
                sources=[p],
                define_macros=define_macros,
            )
        )

    return pkgs, ext_modules

cython_pkgs, cython_ext_modules = get_cython_pkgs_and_ext_modules()

kwargs = {
    "version": "0.1",
    "package_dir": {"": "src"},
}
cython_compiler_directives = {"linetrace": True} if IS_COVERAGE else {}

c_kwargs = {
    "packages": cython_pkgs,
    "ext_modules": cythonize(
        cython_ext_modules,
        language_level="3",
        compiler_directives=cython_compiler_directives,
    ),
    "include_dirs": [np.get_include()],
}
py_kwargs = {
    "packages": find_namespace_packages(where="src"),
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
