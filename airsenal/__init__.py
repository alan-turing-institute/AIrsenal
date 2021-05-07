"""
___init__.py for airsenal
"""
import os
import tempfile

# AIrsenal package version. When merging changes to master:
# - increment 2nd digit for new features
# - increment 3rd digit for bug fixes
__version__ = "0.4.0"

# Cross-platform temporary directory
if os.name == "posix":
    TMPDIR = "/tmp/"
else:
    TMPDIR = tempfile.gettempdir()
