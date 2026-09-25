from setuptools import Extension, setup

import pybind11


ext_modules = [
    Extension(
        "boost_vf2",
        ["boost_vf2.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]


setup(
    name="boost_vf2",
    version="0.0.1",
    ext_modules=ext_modules,
)
