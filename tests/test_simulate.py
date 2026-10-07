import random

import pytest

from src.model import build_model
from src.simulate import BASELINES, hand_value, is_natural, play_hand, simulate
from src.value_iter.value_iteration import solve

N_HANDS = 200_000  # kept small so the tests stay fast

_, OPTIMAL, _ = solve(build_model())


def test_hand_value():
    assert hand_value([1, 10]) == (21, True)
    assert hand_value([1, 1]) == (12, True)
    assert hand_value([1, 6, 10]) == (17, False)   # ace drops to 1
    assert hand_value([10, 6]) == (16, False)


def test_is_natural():
    assert is_natural([1, 10])
    assert not is_natural([1, 4, 6])               # 21, but three cards
    assert not is_natural([10, 9])


def test_rewards_are_valid():
    rng = random.Random(0)
    rewards = {play_hand(OPTIMAL, rng) for _ in range(20_000)}
    assert rewards == {-1.0, 0.0, 1.0, 1.5}


def test_same_seed_gives_same_result():
    assert simulate(OPTIMAL, 10_000, seed=3) == simulate(OPTIMAL, 10_000, seed=3)


def test_rates_sum_to_one():
    r = simulate(OPTIMAL, 10_000)
    assert r["win_rate"] + r["push_rate"] + r["loss_rate"] == pytest.approx(1.0)


def test_optimal_policy_return():
    r = simulate(OPTIMAL, N_HANDS)
    assert r["avg_return"] == pytest.approx(-0.0264, abs=4 * r["std_error"])


@pytest.mark.parametrize("name", BASELINES)
def test_optimal_beats_baselines(name):
    optimal = simulate(OPTIMAL, N_HANDS)["avg_return"]
    baseline = simulate(BASELINES[name], N_HANDS)["avg_return"]
    assert optimal > baseline
