#!/usr/bin/env python

import os
import platform
from glob import glob

from setuptools import find_packages
from numpy.distutils.core import Extension, setup

module_dir = os.path.dirname(os.path.abspath(__file__))


# Utility functions
# -----------------


# def package_files(directory, extensions):
#     """Walk package directory to make sure we include all relevant files in package."""
#     paths = []
#     for (path, directories, filenames) in os.walk(directory):
#         for filename in filenames:
#             if any([filename.endswith(ext) for ext in extensions]):
#                 paths.append(os.path.join("..", path, filename))
#     return paths


# Print the Machine name on screen
# --------------------------------

print("PLATFORM={}".format(platform.platform()))

# Define extension modules
# ------------------------
ext_modules = []

# Call distutils setup
# --------------------

package_name = "horton2_wrapper"
fortran_src_path = os.path.join("horton2_wrapper", "srfunctionals")
src_files = glob("{}/*.F".format(fortran_src_path))
ext_modules = [
    Extension(
        name="{}.srfunctionals.{}".format(package_name, os.path.basename(fname).split(".")[0]),
        sources=[fname],
    )
    for fname in src_files
]

# json_yaml_csv_files = package_files("horton2_wrapper", ["yaml", "json", "csv", "h5", "fchk", "xyz"])

setup(
    name=package_name,
    version="0.0.1",
    description="horton2-wrapper",
    long_description=open(os.path.join(module_dir, "README.md")).read(),
    url="https://github/yingxingcheng/horton2-wrapper",
    author="YingXing Cheng",
    author_email="Yingxing.Cheng@ugent.be",
    license="GNU",
    package_dir={package_name: package_name},
    packages=find_packages(),
    package_data={
        "horton2_wrapper.data": ["*.*"],
        # "horton2_wrapper.other": json_yaml_csv_files,
    },
    # scripts=glob("scripts/*.py"),
    ext_modules=ext_modules,
    zip_safe=False,
    install_requires=[
        "horton>=2.1.0",
        "progress>=1.5",
        "numpy>=1.16.3",
        # "matplotlib>=2.2.5",
        "scipy>=1.2.1",
        # "pandas>=0.24.2",
        # "prettytable>=1.0.1",
        # "sphinx_rtd_theme>=0.4.3",
        "pytest>=4.6.11",
    ],
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
