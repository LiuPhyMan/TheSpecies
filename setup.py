# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = "myspecie",
    packages = find_packages(),
    version = "1.0.0",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "myconst"
    ]
)