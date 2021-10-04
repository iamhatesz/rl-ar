from __future__ import annotations

from typing import List, Union

import gym
import numpy as np
import torch
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper,
    TorchMultiCategorical,
)
from ray.rllib.utils import override
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict

from rl_ar.models.config import DEFAULT_CONFIG
from rl_ar.models.fake_multicategorical import FakeMultiCategoricalModel


class FakeTorchMultiCategorical(TorchMultiCategorical):
    @override(TorchDistributionWrapper)
    def __init__(
        self,
        inputs: List[TensorType],
        model: FakeMultiCategoricalModel,
    ):
        logits_a = model.forward_action_a(inputs)
        logits_b = model.forward_action_b(inputs)
        logits = torch.cat([logits_a, logits_b], dim=-1)
        TorchMultiCategorical.__init__(
            self, logits, model, (model.target, model.target), None
        )

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        config = {**DEFAULT_CONFIG, **model_config.get("custom_model_config", {})}
        return config["num_hiddens"]
