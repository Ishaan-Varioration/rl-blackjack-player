from ..model import GAMMA, HIT, STAND, STATES, TERMINAL, build_model

def q_value (state, action, V, P): # expected return of taking action in state, using the current V for everything after

    q = 0.0 #initialise q value to 0
    for prob, next_state, reward in P[state][action]: #iterating over all possible outcomes for given state and action
        q += prob * (reward + GAMMA * V[next_state]) 
    return q # q = expected value of taking action in state 

P=build_model()

V={state: 0.0 for state in STATES}  # initial value function
V[TERMINAL] = 0.0  # terminal state has value 0

print(q_value((20, False, 6), STAND, V, P))    # expect about  0.678
print(q_value((16, False, 10), HIT, V, P))     # expect about -0.615
