from setuptools import setup

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()
REQUIRED_PACKAGES.append("bpl==v0.0.1")


setup(
    name="airsenal",
    version="0.0.1",
    description="An automatic Fantasy Premier League manager.",
    url="https://github.com/alan-turing-institute/AIrsenal",
    author="Nick Barlow and Angus Williams",
    license="MIT",
    include_package_data=True,
    packages=["airsenal",
              "airsenal.framework",
              "airsenal.scraper",
              "airsenal.scripts"],
    install_requires=REQUIRED_PACKAGES,
    setup_requires=REQUIRED_PACKAGES,
    dependency_links=["https://github.com/anguswilliams91/bpl/archive/v0.0.1-alpha.zip#egg=bpl-v0.0.1"],
    entry_points={"console_scripts": [
            "setup_airsenal_database=airsenal.scripts.fill_db_init:main"
        ],
    },
    package_data={"airsenal": ["data/*", "stan/*"]}
)
