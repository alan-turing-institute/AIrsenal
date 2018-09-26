from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.command.install import install as _install

import os
import subprocess
import sys

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()
SETUP_DIR = os.path.dirname(os.path.abspath(__file__))


class install_(install):
    # inject your own code into this func as you see fit

    def run(self):

        ret = None
        if self.old_and_unmanageable or self.single_version_externally_managed:
            ret = _install.run(self)
        else:
            caller = sys._getframe(2)
            caller_module = caller.f_globals.get('__name__','')
            caller_name = caller.f_code.co_name

            if caller_module != 'distutils.dist' or caller_name!='run_commands':
                _install.run(self)
            else:
                self.do_egg_install()
        sub_path = "{}/bpl".format(SETUP_DIR)
        subprocess.check_call([sys.executable, "-m", "pip", "install", sub_path])
        return ret


setup(
    name="airsenal",
    version="0.0.1",
    description="An automatic Fantasy Premier League manager.",
    url="https://github.com/alan-turing-institute/AIrsenal",
    author="Nick Barlow and Angus Williams",
    license="MIT",
    packages=["airsenal"],
    install_requires=REQUIRED_PACKAGES,
    setup_requires=REQUIRED_PACKAGES,
    cmdclass={"install": install_}
)

