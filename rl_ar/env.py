import random
from typing import Tuple, Optional, TypedDict

import gym
import numpy as np


class SumToTargetEnvConfig(TypedDict):
    target: int


DEFAULT_CONFIG: SumToTargetEnvConfig = {
    "target": 10,
}


class SumToTargetEnv(gym.Env):
    """
    The environment where the objective is to provide two numbers, which summed
    together with the observation result in the target value.

    This is a benchmark for autoregressive models.

    Examples:

        >>> env = SumToTargetEnv({"target": 10})
        >>> env.reset()
        np.ndarray([5])
        >>> env.step((2, 3))
        np.ndarray([5]), 1.0, True, {}

        >>> env = SumToTargetEnv({"target": 20})
        >>> env.reset()
        np.ndarray([11])
        >>> env.step((10, 3))
        np.ndarray([11]), 0.0, True, {}
    """

    def __init__(self, env_config: SumToTargetEnvConfig):
        self._env_config = {**DEFAULT_CONFIG, **env_config}
        self._target = self._env_config["target"]

        self.observation_space = gym.spaces.Box(
            low=0, high=self._target, shape=(1,), dtype=np.float
        )
        self.action_space = gym.spaces.MultiDiscrete([self._target, self._target])

        self._rng = random.Random()

        self._current_obs: Optional[float] = None
        self._last_action: Optional[Tuple[int, int]] = None

    def reset(self) -> np.ndarray:
        self._current_obs = self._rng.randint(0, self._target)
        self._last_action = None
        return np.array([self._current_obs])

    def step(self, action: Tuple[int, int]) -> (np.ndarray, float, bool, dict):
        first, second = action
        self._last_action = action
        total = self._current_obs + first + second
        done = True
        reward = 1.0 if total == self._target else 0.0
        return np.array([self._current_obs]), reward, done, {}

    def render(self, mode="human"):
        if mode == "human":
            print(f"Target: {self._current_obs} | Last action: {self._last_action}")

    def seed(self, seed: Optional[int] = None):
        self._rng.seed(seed)
