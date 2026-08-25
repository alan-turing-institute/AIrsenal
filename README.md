# AIrsenal

![Build Status](https://github.com/alan-turing-institute/AIrsenal/actions/workflows/main.yml/badge.svg)

*AIrsenal* is a package for using Machine learning to pick a Fantasy Premier League team.

## Background and News

For some background information and details see https://www.turing.ac.uk/research/research-programmes/research-engineering/programme-articles/airsenal.

### AIrsenal Details for 2026/27 season

We have made a mini-league **"Prem-AI League"** for players using this software.  To join, login to the FPL website, and navigate to the page to join a league: https://fantasy.premierleague.com/leagues then click "Join a League".
The code to join is: **bancts**.
Hope to see your AI team there!! :)

Our own AIrsenal team's ID for the 2026/27 season is **[1598585](https://fantasy.premierleague.com/entry/1598585/history)**.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for managing Python versions and dependencies. For instructions on how to install uv, go to: https://docs.astral.sh/uv/getting-started/installation/.

### Installation from source [Recommended]

#### Linux and macOS

<details>

**With uv (recommended):**

```shell
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
uv sync
```

**With pip:**

If not using `uv` you can replace `uv sync` with `pip install .` above, but we recommend you do so in a virtual environment, e.g.

```shell
git clone https://github.com/alan-turing-institute/AIrsenal.git
cd AIrsenal
python -m venv .venv
source .venv/bin/activate
pip install .
```

</details>

#### Windows

<details>

The best ways to run AIrsenal on Windows are either to use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) (WSL), which allows you to run AIrsenal in a Linux environment on your Windows system, or Docker (see below).

You can then follow the installation instructions for Linux and macOS above.

Installing AIrsenal on Windows directly is not supported — [jax](https://github.com/google/jax#installation) is the main obstacle. If you get it working we'd love to hear from you.

</details>

#### Docker

<details>

Rather than building and running natively on your machine, you can instead use a Docker image if you prefer.

Build the docker-image:

```console
$ docker build -t airsenal .
```

Create a volume for data persistance:

```console
$ docker volume create airsenal_data
```

Run commands with your configuration as environment variables, eg:

```console
$ docker run -it --rm -v airsenal_data:/tmp/ -e "FPL_TEAM_ID=<your_id>" -e "AIRSENAL_HOME=/tmp" airsenal bash
```

or

```console
$ docker run -it --rm -v airsenal_data:/tmp/ -e "FPL_TEAM_ID=<your_id>" -e "AIRSENAL_HOME=/tmp" airsenal airsenal run
```

`airsenal run` is the default command.

</details>

### AIrsenal on PyPi [Work in Progress]

<details>

⚠️ `pip install airsenal` currently hits dependency issues (see #733), so we recommend [building from source](#installation-from-source) instead. ⚠️

The PyPI release provides the same `airsenal` command described in [Getting Started](#getting-started), but lags behind GitHub.

</details>

### Optional dependencies

<details>

  AIrsenal has optional dependencies for plotting (`plot`), running notebooks (`notebook`), the dev toolchain (`dev`) and the one-off scripts in `tools/` (`tools`). To install them all:
  - With uv: `uv sync --all-extras`
  - Without uv: `pip install ".[notebook,plot,dev,tools]"`

</details>

## Running commands with uv

If using AIrsenal with uv you must either prepend `uv run` to all the AIrsenal commands below (e.g. `uv run airsenal db create`), or activate the virtual environment created by uv and then run them as normal. By default the virtual environment can be activated with `source .venv/bin/activate`.

## Configuration

Once you've installed the module, you will need to set the following parameters:

**Required:**

1. `FPL_TEAM_ID`: the team ID for your FPL side.

**Recommended:**

2. `FPL_LOGIN`: your FPL login, usually email (this is required to get any changes made to your team since the last gameweek deadline).

3. `FPL_PASSWORD`: your FPL password (this is required to get any changes made to your team since the last gameweek deadline).

**Optional:**

4. `FPL_LEAGUE_ID`: a league ID for FPL (this is only required for plotting FPL league standings).

5. `AIRSENAL_DB_FILE`: Local path to where you would like to store the AIrsenal sqlite3 database. If not set `AIRSENAL_HOME/data.db` will be used by default.

The values for these should be defined either in environment variables with the names given above, or as files in `AIRSENAL_HOME` (a directory AIrsenal creates on your system to save config files and the database).

To view the location of `AIRSENAL_HOME` and the current values of all set AIrsenal environment variables run:

```bash
airsenal env get
```

Use `airsenal env set` to set values and store them for future use. For example:

```bash
airsenal env set --key FPL_TEAM_ID --value 123456
```

See `airsenal env --help` for other options.

## Getting Started

**Note:** Most the commands below can be run with the `--help` flag to see additional options and information.

### Run the Full AIrsenal Pipeline

The easiest way to run AIrsenal is to use the pipeline script:

```shell
airsenal run
```

This will create or update the database, compute points predictions, and suggest transfers.  Add `--help` to see the available options, by default predictions and transfers are calculated for the next 3 gameweeks.

Alternatively, you can run each step of AIrsenal independently, as follows:

### 1. Creating the database

Run the following command to create the AIrsenal database:

```shell
airsenal db create
```

This will fill the database with data from the last 3 seasons, as well as all available fixtures and results for the current season.

### 2. Updating the database

Once the database has been created, you just need to update it each time before you run predictions or optimisations. This pulls all the latest data from the FPL API, such as recent match results, changes to fixtures, new players, and player injury/suspension statuses.

```shell
airsenal db update
```

### 3. Running predictions

The next step is to predict the expected points for all players for the next fixtures. Player points predictions are computed using two models, a team-level model to predict match scorelines, and a player level model to predict player goal involvements, as well as several heuristics based on historical averages.

This is done using the command

```shell
airsenal predict --weeks-ahead 3
```

Predicting the next 3 gameweeks of fixtures is the default but this can be configured with the argument above.

### 4. Transfer or Squad Optimization

Finally, we need to run the optimizer to pick the best transfer strategy over the next weeks (and hence the best team for the next week).

```shell
airsenal optimize transfers --weeks-ahead 3
```

This will take a while, but should eventually provide a printout of the optimal transfer strategy, in addition to the teamsheet for the next match (including who to make captain, and the order of the substitutes). You can also optimise chip usage with `--wildcard-week <GW>`, `--free-hit-week <GW>`, `--triple-captain-week <GW>` and `--bench-boost-week <GW>`, replacing `<GW>` with the gameweek you want to play the chip (or `0` to try every gameweek).

Note that `airsenal optimize transfers` should only be used for transfer suggestions after the season has started. If it's before the season has started and you want to generate a full squad for gameweek one you should instead use:

```shell
airsenal optimize squad --num-gameweeks 3
```

### 5. Apply Transfers and Lineup

Note that you must have set `FPL_LOGIN` and `FPL_PASSWORD` for these to work (as described in the "Configuration" section above).

To apply the transfers recommended by AIrsenal to your team on the FPL website run `airsenal apply transfers`.
- **This can't be undone and may incur points hits.** It also can't apply chips such as free hit or wildcard, even if `airsenal optimize transfers` suggested one — play those on the FPL website before the gameweek deadline.

You can also use `airsenal apply lineup` to set your starting lineup, captaincy choices, and substitute order to AIrsenal's recommendation (without making any transfers).

## Issues and New Features

AIrsenal is regularly developed to fix bugs and add new features. If you have any problems during installation or usage please let us know by [creating an issue](https://github.com/alan-turing-institute/AIrsenal/issues/new) (or have a look through [existing issues](https://github.com/alan-turing-institute/AIrsenal/issues) to see if it's something we're already working on).

You may also like to try the development version of AIrsenal, which has the latest fixes and features. To do this checkout the `develop` branch of the repo and reinstall:

```shell
git checkout develop
git pull
uv sync  # or "pip install --force-reinstall ." if not using uv
```

If there have been database changes you may also need to run `airsenal db create --clean` after the above.

## Contributing

We welcome all types of contribution to AIrsenal, for example questions, documentation, bug fixes, new features and more. Please see our [contributing guidelines](CONTRIBUTING.md). If you're contributing for the first time but not sure what to do a good place to start may be to look at our [current issues](https://github.com/alan-turing-institute/AIrsenal/issues), particularly any with the ["Good first issue" tag](https://github.com/alan-turing-institute/AIrsenal/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22). Also feel free to just say hello!


## Development

Install the dev toolchain and the pre-commit hooks, which run formatting, linting, type
checking and the package layering contracts on every commit:

```shell
uv sync --extra dev
pre-commit install --install-hooks
```

Run the tests with:

```shell
uv run pytest tests
```

See [CodingConventions.md](CodingConventions.md) for the conventions we follow, and
[docs/how-it-works.md](docs/how-it-works.md) for the database schema and how points
predictions are built.
