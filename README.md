# AIrsenal
[![Build Status](https://travis-ci.org/alan-turing-institute/AIrsenal.svg?branch=master)](https://travis-ci.org/alan-turing-institute/AIrsenal)

*AIrsenal* is a package for using Machine learning to pick a Fantasy Premier League team.

For some background information and details see https://www.turing.ac.uk/research/research-programmes/research-engineering/programme-articles/airsenal.

We welcome contributions and comments - if you'd like to join the AIrsenal community please refer to our [contribution guidelines](https://github.com/alan-turing-institute/AIrsenal/blob/master/CONTRIBUTING.md)

## Install

It is recommended that you use a conda or virtualenv environment to install and run AIrsenal.  
The Stan model used to predict match results is in the package https://github.com/anguswilliams91/bpl, and to run this you will need a working (recent) C++ compiler.
An example setup could be:
```
conda create -n airsenalenv python=3.7
conda activate airsenalenv
conda install -c psi4 gcc-5 # necessary if you don't have an up-to-date C++ compiler on your system 
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
pip install .
```


## Getting started

Once you've installed the module, you will need to set five environment variables (or alternatively you can put the values into files in the ```airsenal/data/``` directory, e.g. ```airsenal/data/FPL_TEAM_ID```:

1. `FD_API_KEY`: an API key for [football data](https://www.football-data.org/) (this is only needed for filling past seasons results if not already present as a csv file in the ```data/``` directory.)
2. `FPL_TEAM_ID`: the team ID for your FPL side.
3. `FPL_LEAGUE_ID`: a league ID for FPL (this is only required for a small subset of functionality).
4. `FPL_LOGIN`: your FPL login, usually email (this is only required to get league standings)
5. `FPL_PASSWORD`: your FPL password (this is only required to get league standings)

Once this is done, run the following command

```shell
setup_airsenal_database
```

You should get a file ```/tmp/data.db```.  This will fill the database with all that is needed up to the present day.

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





