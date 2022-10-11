from os import path

from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements
from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_reqs = parse_requirements("requirements.txt", session=PipSession())
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name="buildpipe",
    version="0.5.a",
    packages=["buildpipe"],
    url="https://github.com/OldPatrick/buildpipe",
    license="MIT",
    author="bormannp",
    author_email="patrick.bormann@gmx.de",
    description="Validation Pipe for different "
    "experiments based on sklearn architecture with mlflow",
    install_requires=reqs,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
