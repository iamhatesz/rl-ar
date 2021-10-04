from __future__ import annotations

from typing import Union

import gym
import numpy as np
import torch
from ray.rllib.models import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper,
    TorchCategorical,
)
from ray.rllib.utils import override
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict

from rl_ar.models.config import DEFAULT_CONFIG


class FakeAutoregressiveActionDistribution(TorchDistributionWrapper):
    @override(TorchDistributionWrapper)
    def deterministic_sample(self) -> TensorType:
        action_a = self._action_a_dist().deterministic_sample()
        action_b = self._action_b_dist().deterministic_sample()

        sample = torch.stack([action_a, action_b], dim=1)
        self.last_sample = sample
        return sample

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        action_a = self._action_a_dist().sample()
        action_b = self._action_b_dist().sample()

        sample = torch.stack([action_a, action_b], dim=1)
        self.last_sample = sample
        return sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        action_a, action_b = actions[:, 0], actions[:, 1]
        action_logp = self._action_a_dist().logp(action_a) + self._action_b_dist().logp(
            action_b
        )
        return action_logp

    @override(ActionDistribution)
    def multi_entropy(self) -> TensorType:
        entropy_a = self._action_a_dist().entropy()
        entropy_b = self._action_b_dist().entropy()
        return torch.stack([entropy_a, entropy_b], dim=1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return torch.sum(self.multi_entropy(), dim=1)

    @override(ActionDistribution)
    def multi_kl(self, other: FakeAutoregressiveActionDistribution) -> TensorType:
        kl_a = self._action_a_dist().kl(other._action_a_dist())
        kl_b = self._action_b_dist().kl(other._action_b_dist())

        return torch.stack([kl_a, kl_b], dim=1)

    @override(TorchDistributionWrapper)
    def kl(self, other: FakeAutoregressiveActionDistribution) -> TensorType:
        return torch.sum(self.multi_kl(other), dim=1)

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        config = {**DEFAULT_CONFIG, **model_config.get("custom_model_config", {})}
        return config["num_hiddens"]

    def _action_a_dist(self) -> TorchCategorical:
        logits_a = self.model.forward_action_a(self.inputs)
        dist_a = TorchCategorical(logits_a)
        return dist_a

    def _action_b_dist(self) -> TorchCategorical:
        logits_b = self.model.forward_action_b(self.inputs)
        dist_b = TorchCategorical(logits_b)
        return dist_b
