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
from typing import Generic, TypeVar

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
        spec = {f.name: f.type for f in fields(entry.config_cls)}
        unknown = sorted(set(overrides) - set(spec))
        if unknown:
            msg = (
                f"{self._kind} '{name}' has no option(s) {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(spec))}"
            )
            raise ValueError(msg)
        values = {k: _coerce(spec[k], v) for k, v in overrides.items()}
        return entry.factory(entry.config_cls(**values))

    def _lookup(self, name: str) -> _Entry[T]:
        try:
            return self._entries[name]
        except KeyError:
            msg = (
                f"Unknown {self._kind} '{name}'. Choose from: {', '.join(self.names())}"
            )
            raise ValueError(msg) from None


def _coerce(annotation: object, value: str) -> object:
    """Turn a command-line string into the type the config field declares."""
    text = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", str(annotation))
    )
    if "bool" in text:
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        msg = f"expected a boolean, got {value!r}"
        raise ValueError(msg)
    if "int" in text:
        return int(value)
    if "float" in text:
        return float(value)
    return value
