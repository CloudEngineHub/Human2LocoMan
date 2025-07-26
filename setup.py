from setuptools import find_packages
from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='human2locoman',
    version='1.0.0',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='yarun@andrew.cmu.edu',
    description='Human2LocoMan codebase.',
    install_requires=required,
)
