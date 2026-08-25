"""The prediction model tables."""

import inspect

import pytest

from airsenal.core.lookup import ConfigError
from airsenal.prediction.player_models import (
    PLAYER_MODELS,
    ConjugatePlayerConfig,
    ConjugatePlayerModel,
    NumpyroPlayerConfig,
    NumpyroPlayerModel,
    build_player_model,
)
from airsenal.prediction.team_models import (
    DEFAULT_TEAM_MODEL,
    TEAM_MODELS,
    build_team_model,
)


def test_registered_player_models():
    assert sorted(PLAYER_MODELS) == ["conjugate", "constant", "numpyro"]


def test_registered_team_models():
    assert sorted(TEAM_MODELS) == ["constant", "extended", "neutral", "random"]


@pytest.mark.parametrize("name", sorted(PLAYER_MODELS))
def test_every_player_model_implements_the_interface(name):
    model = build_player_model(name)
    assert callable(model.fit)
    assert callable(model.get_probs)


def test_conjugate_is_the_default_player_model():
    model = build_player_model()
    assert isinstance(model, ConjugatePlayerModel)
    assert model.config == ConjugatePlayerConfig()


def test_numpyro_is_selected_by_name_not_a_boolean():
    """--sampling could only ever express two models; a name can express any."""
    assert isinstance(build_player_model("numpyro"), NumpyroPlayerModel)


def test_numpyro_config_has_no_time_weighting_fields():
    assert not hasattr(NumpyroPlayerConfig(), "epsilon")
    assert not hasattr(NumpyroPlayerConfig(), "n_goals_prior")


def test_player_model_fit_takes_no_keyword_arguments():
    """Hyperparameters belong to the model, so fit() has nowhere to drop them."""
    for name in PLAYER_MODELS:
        sig = inspect.signature(build_player_model(name).fit)
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds, name
        assert list(sig.parameters) == ["data"], name


def test_unknown_model_names_list_the_alternatives():
    with pytest.raises(ConfigError, match=r"Unknown player model 'nope'.*conjugate"):
        build_player_model("nope")
    with pytest.raises(ConfigError, match=r"Unknown team model 'nope'.*extended"):
        build_team_model("nope")


def test_a_team_model_holds_the_arguments_it_fits_with():
    """
    bpl takes epsilon when fitting rather than when constructing, so the model
    carries it. Otherwise a caller that builds its own model - replay, say -
    fits with different time weighting than `airsenal run` does.
    """
    model = build_team_model(DEFAULT_TEAM_MODEL)
    assert model.epsilon == 0.9
    assert model.rescale_weights is True


def test_build_team_model_forwards_a_first_class_epsilon():
    assert build_team_model("extended", epsilon=0.5).epsilon == 0.5
    assert build_team_model("neutral", epsilon=0.5).epsilon == 0.5


def test_neutral_and_extended_are_different_bpl_models():
    assert build_team_model("extended").neutral is False
    assert build_team_model("neutral").neutral is True


@pytest.mark.parametrize("name", ["random", "constant"])
def test_a_model_without_time_weighting_rejects_epsilon(name):
    """A model that cannot honour epsilon rejects it rather than ignoring it."""
    build_team_model(name)  # fine without one
    with pytest.raises(ConfigError, match="no time weighting"):
        build_team_model(name, epsilon=0.5)
