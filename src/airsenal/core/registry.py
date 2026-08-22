"""
Name-to-implementation registries.

Model and strategy choices reach the code as strings from the command line. Without
a registry each choice becomes an if/elif chain, and the team-model chain currently
exists twice while the transfer-strategy one is spread over six places. A registry
makes adding an implementation a matter of registering it, and makes an unknown name
an error that lists the valid ones.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class _Entry(Generic[T]):
    factory: Callable[..., T]
    config_cls: type


class Registry(Generic[T]):
    """Maps a name to an implementation and the config dataclass it takes."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, _Entry[T]] = {}

    def register(
        self, name: str, config_cls: type
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Register a factory under a name. Used as a decorator at the definition."""
        if not is_dataclass(config_cls):
            msg = f"config for {self._kind} '{name}' must be a dataclass"
            raise TypeError(msg)

        def decorate(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                msg = f"{self._kind} '{name}' is already registered"
                raise ValueError(msg)
            self._entries[name] = _Entry(factory, config_cls)
            return factory

        return decorate

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def config_cls(self, name: str) -> type:
        return self._lookup(name).config_cls

    def create(self, name: str, config: object | None = None) -> T:
        entry = self._lookup(name)
        return entry.factory(config if config is not None else entry.config_cls())

    def create_with(self, name: str, overrides: Mapping[str, str]) -> T:
        """
        Build from `key=value` strings, as supplied on the command line.

        An unknown key is an error naming the valid ones, rather than being silently
        ignored - which is how player-model hyperparameters came to be quietly
        dropped when the sampling model was selected.
        """
        entry = self._lookup(name)
        try:
            config = self._config_from(entry, overrides)
        except ValueError as e:
            msg = str(e).replace(f"{self._kind} has", f"{self._kind} '{name}' has")
            raise ValueError(msg) from None
        return entry.factory(config)

    def build(
        self, name: str, overrides: Mapping[str, str] | None = None
    ) -> tuple[T, Any]:
        """
        The instance and the config it was built from.

        Some libraries take their settings at fit time rather than construction, so
        the caller needs the config as well as the object. The config type varies
        per registered entry, which one type parameter cannot express, hence Any.
        """
        entry = self._lookup(name)
        config = (
            entry.config_cls() if not overrides else self._config_from(entry, overrides)
        )
        return entry.factory(config), config

    def _config_from(self, entry: "_Entry[T]", overrides: Mapping[str, str]) -> object:
        spec = {f.name: f.type for f in fields(entry.config_cls)}
        unknown = sorted(set(overrides) - set(spec))
        if unknown:
            msg = (
                f"{self._kind} has no option(s) {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(spec))}"
            )
            raise ValueError(msg)
        values = {k: _coerce(spec[k], v) for k, v in overrides.items()}
        return entry.config_cls(**values)

    def _lookup(self, name: str) -> _Entry[T]:
        try:
            return self._entries[name]
        except KeyError:
            msg = (
                f"Unknown {self._kind} '{name}'. Choose from: {', '.join(self.names())}"
            )
            raise ValueError(msg) from None


def _coerce(annotation: object, value: str) -> object:
    """
    Turn a command-line string into the type the config field declares.

    Both spellings of the annotation are inspected: a plain class reports its name
    via __name__, while `float | None` reports "Union" there and only reveals the
    member types via str().
    """
    text = f"{getattr(annotation, '__name__', '')} {annotation}"
    optional = "None" in text or "Optional" in text

    if optional and value.strip().lower() in ("none", "null", ""):
        return None
    if "bool" in text:
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        msg = f"expected a boolean, got {value!r}"
        raise ValueError(msg)
    if "int" in text and "float" not in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value
