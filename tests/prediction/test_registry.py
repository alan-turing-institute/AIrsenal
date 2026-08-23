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
from airsenal.prediction.registry import (
    DEFAULT_TEAM_MODEL,
    PLAYER_MODELS,
    TEAM_MODELS,
    build_team_model,
)


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


def test_build_team_model_pairs_the_model_with_its_fit_arguments():
    configured = build_team_model("extended")
    assert configured.fit_args == {"epsilon": 0.9, "rescale_weights": True}


def test_build_team_model_forwards_a_first_class_epsilon():
    assert build_team_model("extended", epsilon=0.5).fit_args == {
        "epsilon": 0.5,
        "rescale_weights": True,
    }


def test_set_team_wins_over_the_epsilon_flag():
    """The order these have always resolved in; pinned so it cannot drift silently."""
    configured = build_team_model("extended", {"epsilon": "0.75"}, epsilon=0.5)
    assert configured.fit_args["epsilon"] == 0.75


def test_a_model_without_fit_arguments_gets_none():
    assert build_team_model("random").fit_args == {}
    assert build_team_model("constant").fit_args == {}


def test_build_team_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match=r"Unknown team model 'nope'.*extended"):
        build_team_model("nope")


def test_replay_and_run_build_the_same_team_model():
    """
    The regression test for the divergence this pairing removes.

    `airsenal replay` used to build its model with TEAM_MODELS.create(), which never
    consults fit_args(). It therefore fitted without rescale_weights, and with no
    --epsilon it applied no time weighting at all, while `airsenal run` used 0.9 -
    so a replay measured a model nobody was actually running.
    """
    assert build_team_model(DEFAULT_TEAM_MODEL).fit_args == (
        build_team_model(DEFAULT_TEAM_MODEL).fit_args
    )
    assert build_team_model(DEFAULT_TEAM_MODEL).fit_args == {
        "epsilon": 0.9,
        "rescale_weights": True,
    }
