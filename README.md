# AIrsenal
[![Build Status](https://travis-ci.org/alan-turing-institute/AIrsenal.svg?branch=master)](https://travis-ci.org/alan-turing-institute/AIrsenal)

*AIrsenal* is a package for using Machine learning to pick a Fantasy Premier League team.

For some background information and details see https://www.turing.ac.uk/research/research-programmes/research-engineering/programme-articles/airsenal.

We welcome contributions and comments - if you'd like to join the AIrsenal community please refer to our [contribution guidelines](https://github.com/alan-turing-institute/AIrsenal/blob/master/CONTRIBUTING.md)

## Pre-requisites

The Stan model used to predict match results is in the package https://github.com/anguswilliams91/bpl, and to run this you will need a working (recent) C++ compiler. To test you have gcc installed in your system run the following command in a terminal:
```
gcc --version
```

If this successfully returns version information you can continue with the AIrsenal installation process. If not you will need to install `gcc`. Common ways to do this include:
* **Mac OSX:** `brew install gcc`
* **Linux (Ubuntu):** `apt-get install build-essential`
* **Windows:** We recommend using conda and following the windows-specific instructions below. Alternativley, have a look at [MinGW](http://www.mingw.org/wiki/Getting_Started) to get a working compiler.

Alternatively, please refer to the Cython installation pre-requirements for options to get a working compiler on your system here: http://docs.cython.org/en/latest/src/quickstart/install.html.


## Install

We recommend running AIrsenal in a conda environment. For instructions on how to install conda go to this link: https://docs.anaconda.com/anaconda/install/

With conda installed, run these commands in a terminal to create a new conda environment and download and install AIrsenal:

**Linux and Mac OS X:**
```
conda create -n airsenalenv python=3.7 pystan
conda activate airsenalenv
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
pip install .
```

**Windows:**
Based on the pystan instructions here: https://pystan.readthedocs.io/en/latest/windows.html
```
conda create -n airsenalenv python=3.7 pystan
conda activate airsenalenv
conda install libpython m2w64-toolchain -c msys2
conda install numpy cython pystan -c conda-forge
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
pip install .
```

## Getting started

Once you've installed the module, you will need to set some environment variables (or alternatively you can put the values into files in the ```airsenal/data/``` directory, e.g. ```airsenal/data/FPL_TEAM_ID```:

**Required:**
1. `FPL_TEAM_ID`: the team ID for your FPL side.

**Optional:**
2. `FD_API_KEY`: an API key for [football data](https://www.football-data.org/) (this is only needed for filling past seasons results if not already present as a csv file in the ```data/``` directory.)
3. `FPL_LEAGUE_ID`: a league ID for FPL (this is only required for plotting FPL league standings).
4. `FPL_LOGIN`: your FPL login, usually email (this is only required to get FPL league standings).
5. `FPL_PASSWORD`: your FPL password (this is only required to get FPL league standings).

Once this is done, run the following command:

```shell
setup_airsenal_database
```

You should get a file ```/tmp/data.db```.  This will fill the database with all that is needed up to the present day.

You can run sanity checks on the data using the following command:

```
check_airsenal_data
```

## Updating, running predictions and optimization.

To stay up to date in the future, you will need to fill three tables: ```match```, ```player_score```, and ```transaction```
with more recent data, using the command
```shell
update_airsenal_database
```

The next step is to use the team- and player-level Stan models to predict the expected points for all players for the next fixtures.  This is done using the command
```shell
run_airsenal_predictions --weeks_ahead 3
```
(we normally look 3 weeks ahead, as this is an achievable horizon to run the optimization over, but also because things like form and injuries can change a lot in 3 weeks!)

Finally, we need to run the optimizer to pick the best transfer strategy over the next weeks (and hence the best team for the next week).
```shell
run_airsenal_optimization --weeks_ahead 3
```
This will take a while, but should eventually provide a printout of the optimal transfer strategy, in addition to the teamsheet for the next match (including who to make captain, and the order of the substitutes).
