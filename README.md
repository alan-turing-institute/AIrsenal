# AIrsenal

![Build Status](https://github.com/alan-turing-institute/AIrsenal/actions/workflows/main.yml/badge.svg)

*AIrsenal* is a package for using Machine learning to pick a Fantasy Premier League team.

For some background information and details see https://www.turing.ac.uk/research/research-programmes/research-engineering/programme-articles/airsenal.

We welcome contributions and comments - if you'd like to join the AIrsenal community please refer to our [contribution guidelines](https://github.com/alan-turing-institute/AIrsenal/blob/master/CONTRIBUTING.md)

## NEW mini-league for 2021/22 season

We have made a mini-league **"Prem-AI League"** for players using this software.  To join, login to the FPL website, and navigate to the page to join a league: https://fantasy.premierleague.com/leagues/create-join then click "join a league or cup".
The code to join is: **3rn929**.
Hope to see your AI team there! :)

Our own AIrsenal team's id for the 2021/22 season is **[863052](https://fantasy.premierleague.com/entry/863052/history)**.

## Notes (August 2021)

The default branch of this repository has been renamed from "master" to "main".   If you have a previously cloned version of the repo, you can update the branch names by doing:
```
git branch -m master main
git fetch origin
git branch -u origin/main main
git remote set-head origin -a
```

We have also switched from using Stan/pystan to numpyro, for the team and player models.   This will hopefully make AIrsenal easier to install (no need to worry about C compilers any more!).


## Install

We recommend running AIrsenal in a conda environment. For instructions on how to install conda go to this link: https://docs.anaconda.com/anaconda/install/

With conda installed, run these commands in a terminal to create a new conda environment and download and install AIrsenal:

**Linux and Mac OS X:**
```
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
conda env create
conda activate airsenalenv
```

**Windows:**

_Windows is not fully supported. You should be able to install the module but there are still compatibility issues (see issue [#165](https://github.com/alan-turing-institute/AIrsenal/issues/165)). You may have more success trying to run AIrsenal on the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) instead._

If you already have `gcc` working on your system you can follow the Linux & Mac OS X instructions above. Otherwise try the steps below:
```
conda create -n airsenalenv python=3.7
conda activate airsenalenv
conda install libpython m2w64-toolchain -c msys2
conda install numpy cython -c conda-forge
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
pip install .
```

**Use AIrsenal without conda**

To use AIrsenal without conda:
```
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
pip install pygmo  # Linux only
pip install .
```
AIrsenal has an optional optimisation algorithm using the PyGMO package, which is only pip-installable on Linux (either use conda or don't install pygmo on other platforms). However, we have also occasionally seen errors when using conda (e.g. [#81](https://github.com/alan-turing-institute/AIrsenal/issues/81))

**Docker**

Build the docker-image:
```shell
docker build -t airsenal .
```

Create a volume for data persistance:
```shell
docker volume create airsenal_data
```

Run commands with your configuration as environment variables, eg:
```shell
docker run -it --rm -v airsenal_data:/tmp/ -e "FPL_TEAM_ID=<your_id>" airsenal [airsenal_run_pipeline]
```
```airsenal_run_pipeline``` is the default command.


## Configuration

Once you've installed the module, you will need to set the following parameters:

**Required:**
1. `FPL_TEAM_ID`: the team ID for your FPL side.

**Optional:**

2. `FPL_LEAGUE_ID`: a league ID for FPL (this is only required for plotting FPL league standings).

3. `FPL_LOGIN`: your FPL login, usually email (this is only required to get FPL league standings, or automating transfers via the API).

4. `FPL_PASSWORD`: your FPL password (this is only required to get FPL league standings, or automating transfers via the API).

5. `AIrsenalDBFile`: Local path to where you would like to store the AIrsenal sqlite3 database. If not set a temporary directory will be used by default (`/tmp/data.db` on Unix systems).

The values for these should be defined either in environment variables with the names given above, or as files in the `airsenal/data` directory with the names given above. For example, to set your team ID you can create the file `airsenal/data/FPL_TEAM_ID` (with no file extension) and its contents should be your team ID and nothing else. So the contents of the file would just be something like:
```
1234567
```
Where `1234567` is your team ID.

If you do create the files in `airsenal/data`, you should do ```pip install .``` again to ensure they are copied to the correct location for the installed package.

## Getting Started

If you installed AIrsenal with conda, you should always make sure the `airsenalenv` virtual environment is activated before running AIrsenal commands. To activate the environment use:
```shell
conda activate airsenalenv
```

Note: Most the commands below can be run with the `--help` flag to see additional options and information.

### 1. Creating the database

Once the module has been installed and your team ID configured, run the following command to create the AIrsenal database:

```shell
airsenal_setup_initial_db
```
This will fill the database with data from the last 3 seasons, as well as all available fixtures and results for the current season.
On Linux/Mac you should get a file ```/tmp/data.db``` containing the database (on Windows you will get a `data.db` file in a the temporary directory returned by the python [tempfile module](https://docs.python.org/3/library/tempfile.html) on your system).   

You can run sanity checks on the data using the following command:

```
airsenal_check_data
```

### 2. Updating and Running Predictions

To stay up to date in the future, you will need to fill three tables: ```match```, ```player_score```, and ```transaction```
with more recent data, using the command
```shell
airsenal_update_db
```

The next step is to use the team- and player-level Stan models to predict the expected points for all players for the next fixtures.  This is done using the command
```shell
airsenal_run_prediction --weeks_ahead 3
```
(we normally look 3 weeks ahead, as this is an achievable horizon to run the optimization over, but also because things like form and injuries can change a lot in 3 weeks!)

Predicted points must be generated before running the transfer or squad optimization (see below).

### 3. Transfer or Squad Optimization

Finally, we need to run the optimizer to pick the best transfer strategy over the next weeks (and hence the best team for the next week).
```shell
airsenal_run_optimization --weeks_ahead 3
```
This will take a while, but should eventually provide a printout of the optimal transfer strategy, in addition to the teamsheet for the next match (including who to make captain, and the order of the substitutes). You can also optimise chip usage with the arguments ` --wildcard_week <GW>`, `--free_hit_week <GW>`, `--triple_captain_week <GW>` and `--bench_boost_week <GW>`, replacing `<GW>` with the gameweek you want to play the chip (or use `0` to try playing the chip in all gameweeks).

Note that `airsenal_run_optimization` should only be used for transfer suggestions after the season has started. If it's before the season has started and you want to generate a full squad for gameweek one you should instead use:
```shell
airsenal_make_squad --num_gw 3
```
This can also be used during the season to generate a full new squad (e.g. for wildcards).

### 4. Apply Transfers

To apply the transfers recommended by AIrsenal to your team on the FPL website run `airsenal_make_transfers`. This can't be undone! Note that you must have created the `FPL_LOGIN` and `FPL_PASSWORD` files for this to work (as described in the "Configuration" section above).

### Run the Full AIrsenal Pipeline

Instead of running the commands above individually you can use:
```shell
airsenal_run_pipeline
```
This will update the database and then run the points predictions and transfer optimization.  Add `--help` to see the available options.


## Issues and Development

AIrsenal is regularly developed to fix bugs and add new features. If you have any problems during installation or usage please let us know by [creating an issue](https://github.com/alan-turing-institute/AIrsenal/issues/new) (or have a look through [existing issues](https://github.com/alan-turing-institute/AIrsenal/issues) to see if it's something we're already working on).

You may also like to try the development version of AIrsenal, which has the latest fixes and features. To do this checkout the `develop` branch of the repo before the `pip install .` step of the installation instructions above by running the following command:
```
git checkout develop
```

Also, if you wish to make changes to AIrsenal yourself it may be helpful to install AIrsenal in editable mode by adding the `-e` flag to `pip`:
```
pip install -e .
```

## Contributing

We welcome all types of contribution to AIrsenal, for example questions, documentation, bug fixes, new features and more. Please see our [contributing guidelines](CONTRIBUTING.md). If you're contributing for the first time but not sure what to do a good place to start may be to look at our [current issues](https://github.com/alan-turing-institute/AIrsenal/issues), particularly any with the ["Good first issue" tag](https://github.com/alan-turing-institute/AIrsenal/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22). Also feel free to just say hello!

