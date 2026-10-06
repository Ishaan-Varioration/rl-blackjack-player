from .dealer import BUST, add_card, dealer_outcome_dist
from .rules import CARD_PROBS, DEFAULT_RULES

STAND = 0
HIT = 1
TERMINAL = "terminal"  # hand over, value 0
GAMMA = 1.0 #discount factor is 1


def all_states():  
    #this function returns all possible states in the blackjack game, which are represented as tuples of (total, soft, upcard)
    states = []
    for upcard in range(1, 11):
        for total in range(4, 22):       # hard 4..21
            states.append((total, False, upcard))
        for total in range(12, 22):      # soft 12..21
            states.append((total, True, upcard))
    return states


STATES = all_states() # list of all possible states in the blackjack game


def stand_outcomes(state, rules=DEFAULT_RULES):  # win, push, lose
    total, _, upcard = state
    p_win = p_push = p_lose = 0.0
    for dealer_final, p in dealer_outcome_dist(upcard, rules).items():
        if dealer_final == BUST:
            p_win += p
        elif not isinstance(dealer_final, int):  # dealer blackjack (no peek)
            p_lose += p
        elif total > dealer_final:
            p_win += p
        elif total == dealer_final:
            p_push += p
        else:
            p_lose += p
    return [
        (p_win, TERMINAL, 1.0),
        (p_push, TERMINAL, 0.0),
        (p_lose, TERMINAL, -1.0),
    ]


def hit_outcomes(state):  # new state (reward 0) or bust (-1)
    total, soft, upcard = state
    p_bust = 0.0
    p_next = {}
    for card, p in CARD_PROBS.items():
        new_total, new_soft = add_card(total, soft, card)
        if new_total > 21:
            p_bust += p
        else:
            next_state = (new_total, new_soft, upcard)
            p_next[next_state] = p_next.get(next_state, 0.0) + p

    outcomes = [(p, next_state, 0.0) for next_state, p in p_next.items()]
    if p_bust > 0:
        outcomes.append((p_bust, TERMINAL, -1.0))
    return outcomes


def build_model(rules=DEFAULT_RULES):  # P[state][action] = [(prob, next_state, reward)]
    P = {}
    for state in STATES:
        P[state] = {STAND: stand_outcomes(state, rules)}
        if state[0] < 21:  # 21 can only stand
            P[state][HIT] = hit_outcomes(state)
    return P


def _describe(state):
    total, soft, upcard = state
    return f"{'soft' if soft else 'hard'} {total} vs {'A' if upcard == 1 else upcard}"


if __name__ == "__main__":
    P = build_model()
    print(f"{len(STATES)} states")
    for state in [(18, False, 6), (16, False, 10), (17, True, 2)]:
        print(f"\n{_describe(state)}")
        for action, name in ((STAND, "stand"), (HIT, "hit")):
            print(f"  {name}:")
            for p, next_state, reward in P[state][action]:
                target = "hand over" if next_state == TERMINAL else _describe(next_state)
                print(f"    p={p:.3f}  reward={reward:+.0f}  -> {target}")
