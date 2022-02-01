import click
from airsenal.cli import (
    database,
    predict,
    optimise,
    pipeline,
)


@click.group()
def cli():
    pass


@cli.command()
def setup():
    """
    Alias for `airsenal database --setup`.
    """
    database.database.callback(True, False)


@cli.command()
def update():
    """
    Alias for `airsenal database --update`
    """
    database.database.callback(False, True)


cli.add_command(database.database)
cli.add_command(predict.predict)
cli.add_command(optimise.optimise)
cli.add_command(pipeline.run)
