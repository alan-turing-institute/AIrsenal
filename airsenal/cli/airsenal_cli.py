import click
from airsenal.cli import (
    database,
    predict,
    optimise,
)

@click.group()
def cli():
    pass

@cli.command()
def setup():
    database.database.callback(True, False)

@cli.command()
def update():
    database.database.callback(False, True)

cli.add_command(database.database)
# cli.add_command(predict.predict)
# cli.add_command(optimise.optimise)
