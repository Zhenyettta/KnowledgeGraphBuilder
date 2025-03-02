from setuptools import setup, find_packages, Command
import subprocess
from setuptools.command.install import install as _install
from typing import Mapping, Type, cast
import os

with open('requirements.txt', encoding="utf8") as f:
    required = f.read().splitlines()


setup(
    name='kgg',
    version='0.0.1',
    packages=find_packages(include=['kgg', 'kgg.*']),
    install_requires=required
)

os.system('python -m spacy download en_core_web_lg')
