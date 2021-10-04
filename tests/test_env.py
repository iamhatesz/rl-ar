import random

import pytest

from rl_ar.env import SumToTargetEnv


@pytest.mark.parametrize(
    "target",
    [
        10,
        20,
        30,
        40,
        50,
    ],
)
def test_env_rewards_when_obs_and_action_sum_to_target(target: int):
    env = SumToTargetEnv(
        {
            "target": target,
        }
    )
    num_tries = 100
    for _ in range(num_tries):
        obs = env.reset()
        diff = target - int(obs)
        if diff == 0:
            action_a, action_b = 0, 0
        else:
            action_a = random.randint(0, diff - 1)
            action_b = diff - action_a
        new_obs, reward, done, _ = env.step((action_a, action_b))
        assert new_obs == obs
        assert reward == 1.0
        assert done


@pytest.mark.parametrize(
    "target",
    [
        10,
        20,
        30,
        40,
        50,
    ],
)
def test_env_punishes_when_obs_and_action_doesnt_sum_to_target(target: int):
    env = SumToTargetEnv(
        {
            "target": target,
        }
    )
    num_tries = 100
    for _ in range(num_tries):
        obs = env.reset()
        action_a = 1
        action_b = target
        new_obs, reward, done, _ = env.step((action_a, action_b))
        assert new_obs == obs
        assert reward == 0.0
        assert done
