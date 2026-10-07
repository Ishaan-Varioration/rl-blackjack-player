# BLACKJACK PLAYER - REINFORCEMENT LEARNING PROJECT 
## ISHAAN VARIOR AND SEHRAJ MAGU

A Blackjack player built with Dynamic Programming: we model the game as an MDP and solve it with Policy Iteration and Value Iteration.

## Rules
- Infinite deck: each card is drawn independently. A-9 each have probability 1/13, and 10-valued cards (10, J, Q, K) have 4/13
- Player actions: **hit** or **stand**
- Payouts: win +1, loss -1, push 0
- A natural (two-card 21: an ace plus a 10-valued card) pays 1.5x. It is a push if the dealer also has a natural. Any other 21 pays 1x

All rules are configurable in `src/rules.py`.

## Modelling the House
The House operates on a hard and fast rule as it does in standard casinos. 
- Dealer hits on Soft 17 (an ace counted as 11) and below
- Stands on Hard 17 and above, and on Soft 18 and above
- Dealer peeks at face down card to check for blackjack if the face up card is an ace or a 10

Since the dealer's policy is fixed, its behaviour is pure probability. `dealer_outcome_dist(upcard)` in `src/dealer.py` gives the exact probability of the dealer finishing on 17, 18, 19, 20, 21 or busting. Because of the peek, if the player gets to act, the dealer does not have blackjack, so the distribution for an ace or 10 upcard excludes it.

Dealer bust probability by upcard:

| Upcard | A | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Bust % | 20.1 | 35.7 | 37.7 | 39.7 | 41.8 | 43.9 | 26.2 | 24.5 | 22.8 | 23.0 |

## The Model
`src/model.py` turns the game into an MDP. Both algorithms solve this same model, so any difference in their results comes from the algorithms and not from the game.

- **State:** `(player_sum, soft, upcard)`. `soft` means an ace is counted as 11. There are 280 states: hard 4-21 and soft 12-21, against each of the 10 upcards
- **Actions:** `STAND` and `HIT`. A total of 21 can only stand
- **Discount:** gamma = 1, since every hand ends
- **Naturals:** settled on the deal, before any decision, so they are not part of the table

`build_model()` returns one table:

```
P[state][action] = [(probability, next_state, reward), ...]
```

- **Stand** ends the hand. It always has three outcomes: win (+1), push (0) and lose (-1), with probabilities from the dealer model
- **Hit** draws one card. It leads to a new state with reward 0, or to a bust with reward -1

Example, hard 16 against a dealer 10:

| Action | Outcome | Probability | Reward |
|---|---|---|---|
| Stand | Win (dealer busts) | 0.230 | +1 |
| Stand | Push | 0.000 | 0 |
| Stand | Lose | 0.770 | -1 |
| Hit | Hard 17, 18, 19, 20 or 21 | 0.077 each | 0 |
| Hit | Bust | 0.615 | -1 |

## Value Iteration
`src/value_iter/value_iteration.py` finds the optimal policy from the model. It keeps a table `V` with one number per state, the expected return from that state, and improves it until it stops changing.

The update for each state is the Bellman optimality equation:

```
V(s) = max over actions a of   sum of  p * (r + gamma * V(s'))
```

It is built from three functions:
- `q_value(state, action, V, P)`: the expected return of one action in one state. It looks one step ahead and uses the current `V` for everything after
- `sweep(V, P)`: goes through all 280 states once, sets each state's value to the `q_value` of its best action, and returns `delta`, the largest change it made
- `solve(P, theta)`: starts `V` at zero and sweeps until `delta < theta`. It then reads off the policy: the action with the largest `q_value` in each state. Ties go to `STAND`

`solve` returns `V`, `policy` and `stats` (the delta from each sweep, the number of sweeps, the number of state updates and the time taken).

### Convergence
With `theta = 1e-9` it converges in 10 sweeps, which is 2,800 state updates.

| Sweep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| delta | 0.963 | 0.352 | 0.105 | 0.040 | 0.0055 | 0.00056 | 0.000037 | 0.0000015 |

Once it has converged, more sweeps change nothing: the values are optimal for this model.

### The optimal policy
H = hit, S = stand. Rows are the player's total, columns are the dealer's upcard.

```
HARD      2  3  4  5  6  7  8  9  10  A
 4-11     H  H  H  H  H  H  H  H  H   H
 12       H  H  S  S  S  H  H  H  H   H
 13-16    S  S  S  S  S  H  H  H  H   H
 17-21    S  S  S  S  S  S  S  S  S   S

SOFT      2  3  4  5  6  7  8  9  10  A
 12-17    H  H  H  H  H  H  H  H  H   H
 18       S  S  S  S  S  S  S  H  H   H
 19-21    S  S  S  S  S  S  S  S  S   S
```

This matches the standard basic strategy chart for hit and stand.

### The value table
This is the converged `V`. Each number is the expected return per hand, on a bet of 1, from that state when playing optimally. An `h` after a number means the best action is to hit, and no letter means stand. Rows are the player's total, columns are the dealer's upcard, and the `10` column covers all 10-valued cards.

```
HARD        2       3       4       5       6       7       8       9      10       A
  21  +0.880  +0.884  +0.887  +0.891  +0.894  +0.926  +0.931  +0.939  +0.963  +0.904
  20  +0.635  +0.646  +0.657  +0.668  +0.678  +0.773  +0.792  +0.758  +0.555  +0.602
  19  +0.378  +0.397  +0.416  +0.436  +0.453  +0.616  +0.594  +0.288  +0.063  +0.188
  18  +0.110  +0.138  +0.166  +0.195  +0.223  +0.400  +0.106  -0.183  -0.178  -0.226
  17  -0.156  -0.120  -0.083  -0.046  -0.006  -0.107  -0.382  -0.423  -0.420  -0.516
  16  -0.287  -0.247  -0.206  -0.165  -0.121  -0.415h -0.458h -0.509h -0.540h -0.542h
  15  -0.287  -0.247  -0.206  -0.165  -0.121  -0.370h -0.417h -0.472h -0.504h -0.507h
  14  -0.287  -0.247  -0.206  -0.165  -0.121  -0.321h -0.372h -0.431h -0.466h -0.469h
  13  -0.287  -0.247  -0.206  -0.165  -0.121  -0.269h -0.324h -0.387h -0.425h -0.428h
  12  -0.254h -0.234h -0.206  -0.165  -0.121  -0.213h -0.272h -0.340h -0.381h -0.384h
  11  +0.238h +0.260h +0.283h +0.307h +0.332h +0.292h +0.230h +0.158h +0.119h +0.103h
  10  +0.181h +0.205h +0.230h +0.256h +0.282h +0.257h +0.198h +0.117h +0.025h +0.033h
   9  +0.072h +0.099h +0.128h +0.157h +0.187h +0.172h +0.098h -0.052h -0.153h -0.124h
   8  -0.025h +0.006h +0.037h +0.070h +0.104h +0.082h -0.060h -0.210h -0.249h -0.263h
   7  -0.110h -0.077h -0.043h -0.007h +0.030h -0.069h -0.211h -0.285h -0.319h -0.349h
   6  -0.138h -0.105h -0.070h -0.034h +0.005h -0.152h -0.217h -0.293h -0.338h -0.341h
   5  -0.126h -0.093h -0.059h -0.023h +0.015h -0.119h -0.188h -0.267h -0.313h -0.316h
   4  -0.113h -0.081h -0.047h -0.011h +0.026h -0.088h -0.159h -0.241h -0.289h -0.292h

SOFT        2       3       4       5       6       7       8       9      10       A
  21  +0.880  +0.884  +0.887  +0.891  +0.894  +0.926  +0.931  +0.939  +0.963  +0.904
  20  +0.635  +0.646  +0.657  +0.668  +0.678  +0.773  +0.792  +0.758  +0.555  +0.602
  19  +0.378  +0.397  +0.416  +0.436  +0.453  +0.616  +0.594  +0.288  +0.063  +0.188
  18  +0.110  +0.138  +0.166  +0.195  +0.223  +0.400  +0.106  -0.101h -0.144h -0.158h
  17  -0.002h +0.028h +0.059h +0.091h +0.125h +0.054h -0.073h -0.150h -0.197h -0.220h
  16  -0.020h +0.010h +0.041h +0.074h +0.108h -0.005h -0.067h -0.149h -0.207h -0.203h
  15  +0.001h +0.030h +0.060h +0.093h +0.126h +0.037h -0.027h -0.112h -0.174h -0.169h
  14  +0.023h +0.051h +0.081h +0.112h +0.145h +0.080h +0.013h -0.075h -0.139h -0.134h
  13  +0.047h +0.074h +0.103h +0.134h +0.165h +0.122h +0.054h -0.038h -0.105h -0.099h
  12  +0.080h +0.102h +0.127h +0.157h +0.187h +0.165h +0.095h +0.000h -0.070h -0.064h
```

- Hard 13-16 against 2-6 have the same value in each column: the player stands and only wins if the dealer busts, so the exact total does not matter
- Hard 12-16 lose on average whatever the player does
- Soft hands are worth more than the same hard total from 12 to 18, because they can be hit without busting
- Hard and soft 19-21 are identical, since the player stands on both

Print it with `python -m src.value_iter.value_iteration`.

### Checking it by simulation
`src/simulate.py` deals real cards and plays full hands with a given policy, including naturals and the dealer peek. It does not use the model, so it is an independent check. Results over 1,000,000 hands per policy (each average is accurate to about +/- 0.001):

| Policy | Avg return per hand | Win | Push | Lose |
|---|---|---|---|---|
| Value iteration | -0.0275 | 43.1% | 8.7% | 48.1% |
| Mimic dealer (stand on 17+) | -0.0632 | 40.8% | 9.8% | 49.4% |
| Never bust (stand on 12+) | -0.0793 | 41.7% | 6.3% | 51.9% |
| Always stand | -0.1573 | 38.6% | 4.8% | 56.6% |

The expected return predicted from `V` is -0.0264 per hand, which the simulation agrees with. The return is still negative because, with only hit and stand, the house keeps an edge even against perfect play.

## Setup
Requires Python 3.11. Create a virtual environment once, after cloning:
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Run `source .venv/bin/activate` again in each new terminal. The `.venv/` folder is gitignored, so each of us makes our own.

## Running
With the venv active:
```bash
python -m src.dealer            # print the dealer outcome table
python -m src.model             # print example rows of the model
python -m src.value_iter.value_iteration   # run value iteration
python -m src.simulate          # simulate 1,000,000 hands per policy
python -m src.simulate 50000    # same, with a chosen number of hands
pytest                          # run tests
```
