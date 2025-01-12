from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='kgg',
    version='0.0.1',
    packages=find_packages(include=['kgg', 'kgg.*']),
    install_requires=required,
)