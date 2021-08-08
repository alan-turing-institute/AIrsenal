import os.path
import re

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))

# Get package version from airsenal/__init__.py
with open(os.path.join(SETUP_DIR, "airsenal", "__init__.py")) as f:
    VERSION = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

# Get dependencies from requirements.txt
with open(os.path.join(SETUP_DIR, "requirements.txt"), "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()


console_scripts = [
    "airsenal_setup_initial_db=airsenal.scripts.fill_db_init:main",
    "airsenal_update_db=airsenal.scripts.update_db:main",
    "airsenal_plot=airsenal.scripts.plot_league_standings:main",
    "airsenal_run_prediction=airsenal.scripts.fill_predictedscore_table:main",
    "airsenal_run_optimization=airsenal.scripts.fill_transfersuggestion_table:main",
    "airsenal_make_squad=airsenal.scripts.squad_builder:main",
    "airsenal_check_data=airsenal.scripts.data_sanity_checks:run_all_checks",
    "airsenal_dump_db=airsenal.scripts.dump_db_contents:main",
    "airsenal_run_pipeline=airsenal.scripts.airsenal_run_pipeline:run_pipeline",
    "airsenal_replay_season=airsenal.scripts.replay_season:main",
    "airsenal_make_transfers=airsenal.scripts.make_transfers:main",
]

setup(
    name="airsenal",
    version=VERSION,
    description="An automatic Fantasy Premier League manager.",
    url="https://github.com/alan-turing-institute/AIrsenal",
    author="Nick Barlow, Angus Williams, Jack Roberts",
    license="MIT",
    include_package_data=True,
    packages=["airsenal", "airsenal.framework", "airsenal.scripts", "airsenal.api"],
    install_requires=REQUIRED_PACKAGES,
    entry_points={"console_scripts": console_scripts},
    package_data={"airsenal": ["data/*"]},
    zip_safe=False,
)
