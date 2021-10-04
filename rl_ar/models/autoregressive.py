from __future__ import annotations

from typing import Dict, List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict
from torchinfo import summary

from rl_ar.models.config import ModelConfig, DEFAULT_CONFIG


class AutoregressiveModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_config: ModelConfig
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        self._custom_model_config = {**DEFAULT_CONFIG, **custom_model_config}
        self.target = self._custom_model_config["target"]
        self.num_hiddens = self._custom_model_config["num_hiddens"]

        self.features = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, self.num_hiddens),
            nn.Tanh(),
        )
        self.logits_a = nn.Sequential(nn.Linear(self.num_hiddens, self.target))
        self.logits_b = nn.Sequential(
            nn.Linear(self.num_hiddens + self.target, self.num_hiddens),
            nn.Tanh(),
            nn.Linear(self.num_hiddens, self.target),
        )
        self.value = nn.Sequential(
            nn.Linear(self.num_hiddens, 1),
        )

        self._value = None

        print(summary(self))

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"] / self._custom_model_config["target"]
        features = self.extract_features(obs)
        self._value = self.forward_value(features)
        return features, state

    def extract_features(self, obs: TensorType) -> TensorType:
        features = self.features(obs)
        return features

    def forward_action_a(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits_a(features)

    def forward_action_b(
        self, features: torch.Tensor, action_a: torch.Tensor
    ) -> torch.Tensor:
        encoded_action = F.one_hot(action_a, num_classes=self.target)
        final_features = torch.cat((features, encoded_action), dim=-1)
        return self.logits_b(final_features)

    def forward_value(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value(features)
        return value

    def value_function(self) -> TensorType:
        assert self._value is not None
        return torch.reshape(self._value, [-1])
