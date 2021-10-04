import sys
from typing import Optional

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.utils.log import Verbosity

from rl_ar.registry import register_all


def train(config: dict, timesteps_total: int = 50_000, group: Optional[str] = None):
    debug = "pydevd" in sys.modules
    ray.init(local_mode=debug, include_dashboard=False)
    register_all()

    wandb = WandbLoggerCallback(
        project="rl-ar",
        group=group,
        entity="tomasz-wrona-gat",
        api_key_file="~/.wandb",
    )

    default_config = {
        "env": "SumToTargetEnv",
        "env_config": {
            "target": 20,
        },
        "num_workers": 2,
        "num_gpus": 0,
        "framework": "torch",
        "train_batch_size": 100,
        "rollout_fragment_length": 10,
        "sgd_minibatch_size": 10,
        "num_sgd_iter": 15,
        "lr": 0.005,
        "entropy_coeff": 0.01,
        "seed": tune.grid_search([0, 42, 1337, 2137]),
    }

    tune.run(
        "PPO",
        config={**default_config, **config},
        stop={
            "episode_reward_mean": 0.95,
            "timesteps_total": timesteps_total,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        keep_checkpoints_num=3,
        verbose=Verbosity.V3_TRIAL_DETAILS,
        callbacks=[wandb],
    )


def train_baseline():
    config = {
        "model": {
            "custom_model": "BaselineModel",
            "custom_model_config": {
                "target": tune.sample_from(lambda spec: spec.config.env_config.target),
                "num_hiddens": 32,
            },
        },
    }
    train(config, group="baseline")


def train_fake_multicategorical():
    config = {
        "model": {
            "custom_model": "FakeMultiCategoricalModel",
            "custom_model_config": {
                "target": tune.sample_from(lambda spec: spec.config.env_config.target),
                "num_hiddens": 32,
            },
            "custom_action_dist": "FakeTorchMultiCategorical",
        }
    }
    train(config, group="fake_multicategorical")


def train_fake_autoregressive():
    config = {
        "model": {
            "custom_model": "FakeAutoregressiveModel",
            "custom_model_config": {
                "target": tune.sample_from(lambda spec: spec.config.env_config.target),
                "num_hiddens": 32,
            },
            "custom_action_dist": "FakeAutoregressiveActionDistribution",
        }
    }
    train(config, timesteps_total=75_000, group="fake_autoregressive")


def train_autoregressive():
    config = {
        "model": {
            "custom_model": "AutoregressiveModel",
            "custom_model_config": {
                "target": tune.sample_from(lambda spec: spec.config.env_config.target),
                "num_hiddens": 32,
            },
            "custom_action_dist": "AutoregressiveActionDistribution",
        }
    }
    train(config, timesteps_total=75_000, group="autoregressive")


def main():
    train_baseline()
    train_fake_multicategorical()
    train_fake_autoregressive()
    train_autoregressive()


if __name__ == "__main__":
    main()
