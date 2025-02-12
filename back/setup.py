from setuptools import setup, find_packages, Command
import subprocess
from setuptools.command.install import install as _install
from typing import Mapping, Type, cast

with open('requirements.txt', encoding="utf8") as f:
    required = f.read().splitlines()

class CustomInstallCommand(_install):
    def run(self):
        _install.run(self)
        subprocess.check_call(['python', '-m', 'spacy', 'download', 'en_core_web_lg'])

cmdclass: Mapping[str, Type[Command]] = cast(Mapping[str, Type[Command]], {
    'install': CustomInstallCommand
})

setup(
    name='kgg',
    version='0.0.1',
    packages=find_packages(include=['kgg', 'kgg.*']),
    install_requires=required,
    cmdclass=cmdclass
)
