import random
import sys

from .rules import DEFAULT_RULES

N_HANDS = 1_000_000  # default number of simulated hands
STAND = 0
HIT = 1
DECK = list(range(1, 10)) + [10] * 4  # infinite deck: 1 = ace, four 10-valued cards


def hand_value(cards):  # (total, soft), counting one ace as 11 when it fits
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_natural(cards):
    return len(cards) == 2 and hand_value(cards)[0] == 21


def play_hand(policy, rng, rules=DEFAULT_RULES):  # plays one hand, returns the reward
    player = [rng.choice(DECK), rng.choice(DECK)]
    dealer = [rng.choice(DECK), rng.choice(DECK)]
    upcard = dealer[0]

    if is_natural(player):
        return 0.0 if is_natural(dealer) else rules.natural_payout
    if is_natural(dealer) and rules.dealer_peeks:
        return -1.0  # hand ends before the player acts

    while True:  # player's turn
        total, soft = hand_value(player)
        if total > 21:
            return -1.0
        if total == 21 or policy[(total, soft, upcard)] == STAND:
            break
        player.append(rng.choice(DECK))

    if is_natural(dealer):
        return -1.0  # no peek: dealer blackjack found after the player acts

    while True:  # dealer's turn
        dealer_total, dealer_soft = hand_value(dealer)
        if dealer_total > 17 or (dealer_total == 17 and not (dealer_soft and rules.hit_soft_17)):
            break
        dealer.append(rng.choice(DECK))

    if dealer_total > 21 or total > dealer_total:
        return 1.0
    if total == dealer_total:
        return 0.0
    return -1.0


def simulate(policy, n_hands=N_HANDS, seed=0, rules=DEFAULT_RULES):
    rng = random.Random(seed)
    total = total_sq = 0.0
    wins = pushes = losses = 0
    for _ in range(n_hands):
        reward = play_hand(policy, rng, rules)
        total += reward
        total_sq += reward * reward
        if reward > 0:
            wins += 1
        elif reward == 0:
            pushes += 1
        else:
            losses += 1

    mean = total / n_hands
    variance = total_sq / n_hands - mean * mean
    return {
        "hands": n_hands,
        "avg_return": mean,
        "std_error": (variance / n_hands) ** 0.5,  # accuracy of avg_return
        "win_rate": wins / n_hands,
        "push_rate": pushes / n_hands,
        "loss_rate": losses / n_hands,
    }


def threshold_policy(stand_on):  # baseline: stand once the total reaches stand_on
    policy = {}
    for upcard in range(1, 11):
        for total in range(4, 22):
            for soft in (False, True):
                policy[(total, soft, upcard)] = STAND if total >= stand_on else HIT
    return policy


BASELINES = {
    "mimic dealer (stand on 17+)": threshold_policy(17),
    "never bust (stand on 12+)": threshold_policy(12),
    "always stand": threshold_policy(4),
}


if __name__ == "__main__":
    from .model import build_model
    from .value_iter.value_iteration import solve

    n_hands = int(sys.argv[1]) if len(sys.argv) > 1 else N_HANDS
    _, optimal, _ = solve(build_model())

    print(f"{n_hands:,} hands per policy\n")
    print(f"{'policy':<30}{'avg return':>12}{'+/-':>9}{'win':>8}{'push':>8}{'lose':>8}")
    for name, policy in [("value iteration", optimal)] + list(BASELINES.items()):
        r = simulate(policy, n_hands)
        print(
            f"{name:<30}{r['avg_return']:>+12.4f}{r['std_error']:>9.4f}"
            f"{r['win_rate']:>8.1%}{r['push_rate']:>8.1%}{r['loss_rate']:>8.1%}"
        )
