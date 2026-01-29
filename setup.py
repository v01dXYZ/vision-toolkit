from setuptools import Extension, setup, find_packages
import numpy as np
from Cython.Build import cythonize 

setup(
     name="vision_toolkit",
     version="0.1",
     packages=find_packages(where="src"),
     package_dir={"": "src"}, 
)

