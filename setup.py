#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="rtfm",
    version="0.0",
    description="LLMs for tabular data.",
    author="Josh Gardner",
    author_email="jpgard@cs.washington.edu",
    packages=find_packages(),
)
