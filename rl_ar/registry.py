from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from rl_ar.action_dists.fake_autoregressive import FakeAutoregressiveActionDistribution
from rl_ar.action_dists.autoregressive import AutoregressiveActionDistribution
from rl_ar.action_dists.fake_multicategorical import FakeTorchMultiCategorical
from rl_ar.env import SumToTargetEnv
from rl_ar.models.autoregressive import AutoregressiveModel
from rl_ar.models.baseline import BaselineModel
from rl_ar.models.fake_autoregressive import FakeAutoregressiveModel
from rl_ar.models.fake_multicategorical import FakeMultiCategoricalModel


def register_envs():
    register_env("SumToTargetEnv", lambda env_config: SumToTargetEnv(env_config))


def register_models():
    ModelCatalog.register_custom_model("BaselineModel", BaselineModel)
    ModelCatalog.register_custom_model(
        "FakeMultiCategoricalModel", FakeMultiCategoricalModel
    )
    ModelCatalog.register_custom_model(
        "FakeAutoregressiveModel", FakeAutoregressiveModel
    )
    ModelCatalog.register_custom_model("AutoregressiveModel", AutoregressiveModel)


def register_action_dists():
    ModelCatalog.register_custom_action_dist(
        "FakeTorchMultiCategorical", FakeTorchMultiCategorical
    )
    ModelCatalog.register_custom_action_dist(
        "FakeAutoregressiveActionDistribution", FakeAutoregressiveActionDistribution
    )
    ModelCatalog.register_custom_action_dist(
        "AutoregressiveActionDistribution", AutoregressiveActionDistribution
    )


def register_all():
    register_envs()
    register_models()
    register_action_dists()
