import os
import pathlib
import numpy as np

from setuptools import Extension, setup, find_packages, find_namespace_packages
from Cython.Build import cythonize


def parse_boolean_env_var(key):
    value = os.getenv(key, "").lower()
    if value in ("1", "true"):
        return True
    elif value in ("0", "false"):
        return False

    return None

VISION_TOOLKIT_COVERAGE = parse_boolean_env_var("VISION_TOOLKIT_COVERAGE")
VISION_TOOLKIT_CYTHON_CACHE = parse_boolean_env_var("VISION_TOOLKIT_CYTHON_CACHE")

setup(
    ext_modules= cythonize(
        Extension(
            "*",
            ["src/**/*.pyx"],
            define_macros=[
                ("CYTHON_TRACE_NOGIL", "1"),
                # ("CYTHON_TRACE", "1"), # the macro above implies this one as well
                # ---
                # As of 2026, coverage has a "sysmon" core (for sys.monitoring)
                # BUT Plugin filetracers are not supported with SysMonitoring.
                # Since Cython default to sys.monitoring with 3.13+,
                # we explicitly have to set it to use the old sys.settrace.
                ("CYTHON_USE_SYS_MONITORING", "0"),
            ] if VISION_TOOLKIT_COVERAGE else [],
        ),
        language_level=3,
        compiler_directives={"linetrace": True} if VISION_TOOLKIT_COVERAGE else {},
        cache=VISION_TOOLKIT_CYTHON_CACHE,
    ),
    include_dirs= [np.get_include()],
)
