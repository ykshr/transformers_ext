#!/usr/bin/env python3
from setuptools import find_packages, setup

def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name="transformersExt",
    version="0.1",
    author="ykshr",
    url="https://github.com/ykshr/transformers_ext",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=_requires_from_file('requirements.txt'),
)
