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
        "setup_airsenal_database=airsenal.scripts.fill_db_init:main",
        "update_airsenal_database=airsenal.scripts.update_results_transactions_db:main",
        "airsenal_plot=airsenal.scripts.plot_league_standings:main",
        "run_airsenal_predictions=airsenal.scripts.fill_predictedscore_table:main",
        "run_airsenal_optimization=airsenal.scripts.fill_transfersuggestion_table:main",
        "airsenal_make_team=airsenal.scripts.team_builder:main",
        "check_airsenal_data=airsenal.scripts.data_sanity_checks:run_all_checks",
        "dump_db_contents=airsenal.scripts.dump_db_contents:main"
        ],

    },
    package_data={"airsenal": ["data/*", "stan/*"]},
)
