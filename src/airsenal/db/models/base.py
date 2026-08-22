"""Declarative base and shared column type annotations."""

from typing import Annotated

from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, mapped_column

# Common type annotations using PEP 593 Annotated
intpk = Annotated[int, mapped_column(primary_key=True)]
str100 = Annotated[str, mapped_column(String(100))]
str4 = Annotated[str, mapped_column(String(4))]
str3 = Annotated[str, mapped_column(String(3))]
str100_optional = Annotated[str | None, mapped_column(String(100))]


class Base(DeclarativeBase):
    pass
