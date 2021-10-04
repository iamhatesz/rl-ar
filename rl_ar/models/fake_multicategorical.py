from __future__ import annotations

from typing import Dict, List

from ray.rllib.utils.framework import TensorType

from rl_ar.models.baseline import BaselineModel


class FakeMultiCategoricalModel(BaselineModel):
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
