"""The prediction model registries."""

import inspect

import pytest

from airsenal.prediction.config import (
    ConjugatePlayerConfig,
    DixonColesConfig,
    NumpyroPlayerConfig,
)
from airsenal.prediction.player_models import (
    BasePlayerModel,
    ConjugatePlayerModel,
    NumpyroPlayerModel,
)
from airsenal.prediction.registry import PLAYER_MODELS, TEAM_MODELS


def test_registered_player_models():
    assert PLAYER_MODELS.names() == ("conjugate", "constant", "numpyro")


def test_registered_team_models():
    assert TEAM_MODELS.names() == ("constant", "extended", "neutral", "random")


@pytest.mark.parametrize("name", PLAYER_MODELS.names())
def test_every_player_model_implements_the_interface(name):
    assert isinstance(PLAYER_MODELS.create(name), BasePlayerModel)


def test_conjugate_is_the_default_player_model():
    model = PLAYER_MODELS.create("conjugate")
    assert isinstance(model, ConjugatePlayerModel)
    assert model.config == ConjugatePlayerConfig()


def test_numpyro_is_selected_by_name_not_a_boolean():
    """--sampling could only ever express two models; a name can express any."""
    assert isinstance(PLAYER_MODELS.create("numpyro"), NumpyroPlayerModel)


def test_player_model_options_reach_the_model():
    model = PLAYER_MODELS.create_with(
        "conjugate", {"epsilon": "0.5", "n_goals_prior": "7"}
    )
    assert model.config.epsilon == 0.5
    assert model.config.n_goals_prior == 7


def test_asking_numpyro_for_epsilon_is_an_error():
    """
    Regression test for the silent-drop bug: fit() took **kwargs, so epsilon was
    accepted and ignored by the sampling model, quietly disabling time weighting.
    """
    with pytest.raises(ValueError, match=r"no option\(s\) epsilon"):
        PLAYER_MODELS.create_with("numpyro", {"epsilon": "0.2"})


def test_numpyro_config_has_no_time_weighting_fields():
    assert not hasattr(NumpyroPlayerConfig(), "epsilon")
    assert not hasattr(NumpyroPlayerConfig(), "n_goals_prior")


def test_player_model_fit_takes_no_keyword_arguments():
    """Hyperparameters belong to the model, so fit() has nowhere to drop them."""
    for name in PLAYER_MODELS.names():
        sig = inspect.signature(PLAYER_MODELS.create(name).fit)
        kinds = {p.kind for p in sig.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds, name
        assert list(sig.parameters) == ["data"], name


def test_unknown_model_names_list_the_alternatives():
    with pytest.raises(ValueError, match=r"Unknown player model 'nope'.*conjugate"):
        PLAYER_MODELS.create("nope")
    with pytest.raises(ValueError, match=r"Unknown team model 'nope'.*extended"):
        TEAM_MODELS.create("nope")


def test_build_returns_the_config_alongside_the_team_model():
    """bpl takes epsilon when fitting, so the caller needs the config too."""
    _model, config = TEAM_MODELS.build("extended", {"epsilon": "0.75"})
    assert config == DixonColesConfig(epsilon=0.75)
    assert config.fit_args() == {"epsilon": 0.75, "rescale_weights": True}


def test_random_team_model_needs_no_fit_arguments():
    _model, config = TEAM_MODELS.build("random")
    assert config.fit_args() == {}
