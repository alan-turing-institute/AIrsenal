import os.path
import pickle

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SETUP_DIR, "stan")
MODEL_TARGET_DIR = os.path.join("airsenal", "stan_model")


class BPyCmd(build_py):
    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            compile_stan_models(target_dir)

        build_py.run(self)


class DevCmd(develop):
    def run(self):
        if not self.dry_run:
            target_dir = os.path.join(self.setup_path, MODEL_TARGET_DIR)
            self.mkpath(target_dir)
            compile_stan_models(target_dir)

        develop.run(self)


def compile_stan_models(target_dir, model_dir=MODEL_DIR):
    """Pre-compile the stan models that are used by the module."""
    from pystan import StanModel

    print("Compiling Stan player model, and putting pickle in {}".format(target_dir))
    sm = StanModel(file=os.path.join(model_dir, "player_forecasts.stan"))
    with open(os.path.join(target_dir, "player_forecasts.pkl"), "wb") as f_stan:
        pickle.dump(sm, f_stan, protocol=pickle.HIGHEST_PROTOCOL)


setup(
    name="airsenal",
    version="0.1.0",
    description="An automatic Fantasy Premier League manager.",
    url="https://github.com/alan-turing-institute/AIrsenal",
    author="Nick Barlow, Angus Williams, Jack Roberts",
    license="MIT",
    include_package_data=True,
    packages=["airsenal", "airsenal.framework", "airsenal.scripts", "airsenal.api"],
    install_requires=REQUIRED_PACKAGES,
    entry_points={
        "console_scripts": [
            "airsenal_setup_initial_db=airsenal.scripts.fill_db_init:main",
            "airsenal_update_db=airsenal.scripts.update_db:main",
            "airsenal_plot=airsenal.scripts.plot_league_standings:main",
            "airsenal_run_prediction=airsenal.scripts.fill_predictedscore_table:main",
            "airsenal_run_optimization=airsenal.scripts.fill_transfersuggestion_table:main",
            "airsenal_make_squad=airsenal.scripts.squad_builder:main",
            "airsenal_check_data=airsenal.scripts.data_sanity_checks:run_all_checks",
            "airsenal_dump_db=airsenal.scripts.dump_db_contents:main",
            "airsenal_run_pipeline=airsenal.scripts.airsenal_run_pipeline:airsenal_run_pipeline",
            "airsenal_replay_season=airsenal.scripts.replay_season:main",
        ]
    },
    package_data={"airsenal": ["data/*"]},
    zip_safe=False,
    cmdclass={"build_py": BPyCmd, "develop": DevCmd},
)
