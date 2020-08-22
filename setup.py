from setuptools import setup

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()


setup(
    name="airsenal",
    version="0.0.1",
    description="An automatic Fantasy Premier League manager.",
    url="https://github.com/alan-turing-institute/AIrsenal",
    author="Nick Barlow and Angus Williams",
    license="MIT",
    include_package_data=True,
    packages=["airsenal", "airsenal.framework", "airsenal.scraper", "airsenal.scripts"],
    install_requires=REQUIRED_PACKAGES,

    entry_points={"console_scripts": [
        "airsenal_setup_initial_db=airsenal.scripts.fill_db_init:main",
        "airsenal_update_db=airsenal.scripts.update_results_transactions_db:main",
        "airsenal_plot=airsenal.scripts.plot_league_standings:main",
        "airsenal_run_prediction=airsenal.scripts.fill_predictedscore_table:main",
        "airsenal_run_optimization=airsenal.scripts.fill_transfersuggestion_table:main",
        "airsenal_make_team=airsenal.scripts.team_builder:main",
        "airsenal_check_data=airsenal.scripts.data_sanity_checks:run_all_checks",
        "airsenal_dump_db=airsenal.scripts.dump_db_contents:main",
        "airsenal_run_pipeline=airsenal.scripts.airsenal_run_pipeline:airsenal_run_pipeline"
        ],

    },
    package_data={"airsenal": ["data/*", "stan/*"]},
)
