import pytest

from src.model import HIT, STAND, STATES, TERMINAL, build_model
from src.rules import Rules

P = build_model()


def test_state_count():
    assert len(STATES) == 280
    assert len(set(STATES)) == 280


@pytest.mark.parametrize("rules", [Rules(), Rules(hit_soft_17=False, dealer_peeks=False)])
def test_probabilities_sum_to_one(rules):
    for actions in build_model(rules).values():
        for outcomes in actions.values():
            assert sum(p for p, _, _ in outcomes) == pytest.approx(1.0)


def test_next_states_are_valid():
    valid = set(STATES) | {TERMINAL}
    for actions in P.values():
        for outcomes in actions.values():
            for _, next_state, _ in outcomes:
                assert next_state in valid


def test_only_stand_at_21():
    for state, actions in P.items():
        expected = {STAND} if state[0] == 21 else {STAND, HIT}
        assert set(actions) == expected


def test_stand_always_has_win_push_lose():
    for actions in P.values():
        assert [(s, r) for _, s, r in actions[STAND]] == [
            (TERMINAL, 1.0),
            (TERMINAL, 0.0),
            (TERMINAL, -1.0),
        ]


def test_cannot_bust_below_12_or_when_soft():
    for (total, soft, _), actions in P.items():
        if HIT in actions and (total < 12 or soft):
            assert all(s != TERMINAL for _, s, _ in actions[HIT])


def test_hit_keeps_the_upcard():
    for state, actions in P.items():
        for _, next_state, _ in actions.get(HIT, []):
            if next_state != TERMINAL:
                assert next_state[2] == state[2]


def test_stand_hard_18_vs_6():
    p_win, p_push, p_lose = (p for p, _, _ in P[(18, False, 6)][STAND])
    assert p_win == pytest.approx(0.554, abs=2e-3)
    assert p_push == pytest.approx(0.115, abs=2e-3)
    assert p_lose == pytest.approx(0.331, abs=2e-3)


def test_hit_hard_16():
    outcomes = {s: (p, r) for p, s, r in P[(16, False, 10)][HIT]}
    assert outcomes[TERMINAL] == (pytest.approx(8 / 13), -1.0)   # 6..10 bust
    assert outcomes[(17, False, 10)] == (pytest.approx(1 / 13), 0.0)
    assert len(outcomes) == 6                                     # 17..21 + bust


def test_hit_soft_17_ten_makes_hard_17():
    outcomes = {s: p for p, s, _ in P[(17, True, 2)][HIT]}
    assert outcomes[(17, False, 2)] == pytest.approx(4 / 13)
    assert outcomes[(21, True, 2)] == pytest.approx(1 / 13)       # drew a 4
