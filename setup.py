#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

PACKAGE_NAME = 'aics_tf_registration'

setup_requirements = [
    "wheel>=0.34.2",
    "setuptools",
    'scikit-image>=0.16.2',
    'aicsimageio>=3.2.0',
    'SimpleITK>=1.2.4',
    'numpy>=1.17.3',
    'scipy>=1.3.1',
    'pandas>=0.25.2',
    'tqdm>=4.36.1',
    'ruamel.yaml',
    # 'tornado>=6.0.3'
    # 'pyyaml>5.1.2'
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bumpversion>=0.6.0",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r>=0.2.1",
    "pytest-runner>=5.2",
    "Sphinx>=2.0.0b1,<3",
    "sphinx_rtd_theme>=0.4.3",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    setup_requirements,
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
        *test_requirements,
    ]
}

setup(
    author="Mark Filip Sluzewski",
    author_email="filip.sluzewski@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Rigid registration algorithm for generating training/testing data for transfer function model", # noqa
    entry_points={
        "console_scripts": [
            "run_alignment={}.bin.run_alignment:main".format(PACKAGE_NAME)
        ]
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="aics_tf_registration",
    name="aics_tf_registration",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.7",
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url="https://github.com/aics-int/aics_tf_registration",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="1.0.0",
    zip_safe=False,
)