from setuptools import setup, find_packages
import os


__version__ = 0.01

packages = find_packages(exclude=["docs", "notebooks", "assets"])

#
# Base installation (interface only)
#
install_requires = [
    "rq",
    "torch",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="rlberry-farms",
    version=__version__,
    description="some farms for rlberry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Timothée Mathieu",
    url="https://gitlab.inria.fr/scool/rlberry-farms",
    license="MIT",
    packages=packages,
    install_requires=install_requires,
    zip_safe=False,
)
