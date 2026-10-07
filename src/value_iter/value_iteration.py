import time
from ..model import GAMMA, HIT, STAND, STATES, TERMINAL, build_model

def q_value (state, action, V, P): # expected return of taking action in state, using the current V for everything after

    q = 0.0 #initialise q value to 0
    for prob, next_state, reward in P[state][action]: #iterating over all possible outcomes for given state and action
        q += prob * (reward + GAMMA * V[next_state]) 
    return q # q = expected value of taking action in state 

def sweep(V,P): #goes through all states and updates the value function based on the expected return of taking the best action in each state

    delta = 0.0 
    for state in STATES:
        v = V[state] 
        V[state] = max(q_value(state, action, V, P) for action in P[state]) # update V[STATE] to max q value possible from that state
        delta = max(delta, abs(v - V[state]))
    return delta #ends up being the maximum change in value function across all states

# a large delta value means estimates are still being corrected 
# a small delta value means estimates are converging to the true value function


def solve(P, theta=1e-9): # sweeps until the value function converges
    #theta = 10^-9

    start = time.perf_counter()  # start timer
    V = {state: 0.0 for state in STATES}  # initial value function
    V[TERMINAL] = 0.0  # terminal state has value 0

    deltas = []  # delta from each sweep, for the convergence plot
    while True:
        delta = sweep(V, P)
        deltas.append(delta)
        if delta < theta:  # nothing changed by more than theta, so we have converged
            break

    policy = {}  # best action in each state
    for state in STATES:
        policy[state] = max(P[state], key=lambda action: q_value(state, action, V, P))

    stats = {
        "deltas": deltas,
        "sweeps": len(deltas),
        "time": time.perf_counter() - start,
        "updates": len(deltas) * len(STATES),
    }
    return V, policy, stats


if __name__ == "__main__":
    V, policy, stats = solve(build_model())
    print(stats["sweeps"], "sweeps")
    print(stats["updates"], "updates")
    print(round(stats["time"], 4), "seconds")


