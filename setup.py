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
    packages=["airsenal"],
    install_requires=REQUIRED_PACKAGES
)