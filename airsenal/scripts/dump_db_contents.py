"""
Script to dump the database contents.
"""

import csv
import os

from sqlalchemy import inspect

from airsenal.framework.schema import Base
from airsenal.framework.utils import session


def main():
    dump_path = os.path.join(os.path.dirname(__file__), "../data/db_dump")
    os.makedirs(dump_path, exist_ok=True)

    tables = dict(Base.metadata.tables)
    for name, dbclass in tables.items():
        print(dbclass, type(dbclass))
        print("dumping table: ", name)
        save_table_fields(dump_path, dbclass, name)


def save_table_fields(path, dbclass, tablename):
    result = os.path.join(path, f"{tablename}.csv")
    with open(result, "w") as csvfile:
        write_rows_to_csv(csvfile, dbclass)

    return result


def write_rows_to_csv(csvfile, dbclass):
    fieldnames = [col.name for col in inspect(dbclass).columns]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for player in session.query(dbclass).all():
        print(player, dbclass)
        player = vars(player)
        row = {
            field: player[field]
            for field, value____ in player.items()
            if isinstance(value____, (str, int, float))
        }

        writer.writerow(row)


if __name__ == "__main__":
    print(" ==== dumping database contents === ")
    main()
