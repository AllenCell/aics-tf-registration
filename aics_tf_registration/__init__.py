# -*- coding: utf-8 -*-

"""Top-level package for AICS Transfer Function Registration."""

__author__ = "Mark Filip Sluzewski"
__email__ = "filip.sluzewski@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "1.0.0"


def get_module_version():
    return __version__


from .example import Example  # noqa: F401
