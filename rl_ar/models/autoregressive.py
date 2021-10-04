from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_ar.models.fake_autoregressive import FakeAutoregressiveModel


class AutoregressiveModel(FakeAutoregressiveModel):
    def forward_action_b(
        self, features: torch.Tensor, action_a: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert action_a is not None
        encoded_action = F.one_hot(action_a, num_classes=self.target)
        final_features = torch.cat((features, encoded_action), dim=-1)
        return self.logits_b(final_features)

    def _init_logits_b(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.num_hiddens + self.target, self.num_hiddens),
            nn.Tanh(),
            nn.Linear(self.num_hiddens, self.target),
        )
