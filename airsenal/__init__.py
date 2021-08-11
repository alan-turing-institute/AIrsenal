"""
___init__.py for airsenal
"""

import os
import tempfile

# AIrsenal package version.
__version__ = "1.0.0"

# Cross-platform temporary directory
TMPDIR = "/tmp/" if os.name == "posix" else tempfile.gettempdir()
