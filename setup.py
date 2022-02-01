import os.path
import re

from setuptools import setup

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))

# Get package version from airsenal/__init__.py
with open(os.path.join(SETUP_DIR, "airsenal", "__init__.py")) as f:
    VERSION = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

# Get dependencies from requirements.txt
with open(os.path.join(SETUP_DIR, "requirements.txt"), "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()


console_scripts = [
    "airsenal_plot=airsenal.scripts.plot_league_standings:main",
    "airsenal_replay_season=airsenal.scripts.replay_season:main",
    "airsenal_cli=airsenal.cli.airsenal_cli:cli",
]

setup(
    name="airsenal",
    version=VERSION,
    description="An automatic Fantasy Premier League manager.",
    url="https://github.com/alan-turing-institute/AIrsenal",
    author="Nick Barlow, Angus Williams, Jack Roberts",
    license="MIT",
    include_package_data=True,
    packages=[
        "airsenal",
        "airsenal.framework",
        "airsenal.scripts",
        "airsenal.api",
        "airsenal.cli",
    ],
    install_requires=REQUIRED_PACKAGES,
    entry_points={"console_scripts": console_scripts},
    package_data={"airsenal": ["data/*"]},
    zip_safe=False,
)
