"""The name-to-implementation registry."""

from dataclasses import dataclass

import pytest

from airsenal.core.registry import Registry


@dataclass(frozen=True)
class DummyConfig:
    epsilon: float | None = 0.5
    n_prior: int = 10
    rescale: bool = True
    label: str = "default"


class Dummy:
    def __init__(self, config: DummyConfig) -> None:
        self.config = config


@pytest.fixture
def registry():
    reg: Registry[Dummy] = Registry("dummy model")

    @reg.register("basic", DummyConfig)
    def _make(config: DummyConfig) -> Dummy:
        return Dummy(config)

    return reg


def test_create_uses_the_config_defaults(registry):
    assert registry.create("basic").config == DummyConfig()


def test_create_accepts_an_explicit_config(registry):
    config = DummyConfig(epsilon=0.9)
    assert registry.create("basic", config).config is config


def test_names_are_sorted(registry):
    registry.register("another", DummyConfig)(Dummy)
    assert registry.names() == ("another", "basic")


def test_unknown_name_lists_the_valid_ones(registry):
    with pytest.raises(ValueError, match=r"Unknown dummy model 'nope'.*basic"):
        registry.create("nope")


def test_registering_the_same_name_twice_is_an_error(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register("basic", DummyConfig)(Dummy)


def test_config_must_be_a_dataclass():
    reg: Registry[Dummy] = Registry("dummy model")
    with pytest.raises(TypeError, match="must be a dataclass"):
        reg.register("bad", dict)(Dummy)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"epsilon": "0.25"}, DummyConfig(epsilon=0.25)),
        ({"n_prior": "42"}, DummyConfig(n_prior=42)),
        ({"rescale": "false"}, DummyConfig(rescale=False)),
        ({"rescale": "0"}, DummyConfig(rescale=False)),
        ({"rescale": "yes"}, DummyConfig(rescale=True)),
        ({"label": "custom"}, DummyConfig(label="custom")),
    ],
)
def test_create_with_coerces_command_line_strings(registry, overrides, expected):
    assert registry.create_with("basic", overrides).config == expected


def test_create_with_rejects_an_unknown_option(registry):
    """
    The bug this prevents: NumpyroPlayerModel.fit swallowed epsilon in **kwargs, so
    asking for a hyperparameter it does not implement silently did nothing.
    """
    with pytest.raises(ValueError, match=r"no option\(s\) sampling.*epsilon"):
        registry.create_with("basic", {"sampling": "true"})


def test_create_with_reports_a_bad_value(registry):
    with pytest.raises(ValueError, match="expected a boolean"):
        registry.create_with("basic", {"rescale": "maybe"})


def test_optional_float_is_coerced_not_left_as_a_string(registry):
    """
    Regression test. The coercion looked at __name__, which for `float | None` is
    "Union" on 3.14, so the value came through as the string "0.25" and every
    downstream arithmetic operation would have been wrong.
    """
    config = registry.create_with("basic", {"epsilon": "0.25"}).config
    assert config.epsilon == 0.25
    assert isinstance(config.epsilon, float)


@pytest.mark.parametrize("value", ["none", "None", "null", ""])
def test_optional_field_accepts_none(registry, value):
    assert registry.create_with("basic", {"epsilon": value}).config.epsilon is None


def test_non_optional_field_does_not_accept_none(registry):
    with pytest.raises(ValueError, match=r"could not convert|invalid literal"):
        registry.create_with("basic", {"n_prior": "none"})
