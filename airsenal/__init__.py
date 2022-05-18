"""
___init__.py for airsenal
"""

import os
import tempfile

import pkg_resources

# AIrsenal package version.
__version__ = pkg_resources.get_distribution("airsenal").version

# Cross-platform temporary directory
TMPDIR = "/tmp/" if os.name == "posix" else tempfile.gettempdir()
