import gym
from collections import defaultdict
import plot

class BlackjackModified:
    def __init__(self):
        self._env = gym.make("Blackjack-v0")

    def reset(self):
        return self._env.reset()

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        if done == True:
            return 'TERMINAL', reward, done
        else:
            return next_state, reward, done


env = BlackjackModified()

def td_evaluation(n_episodes, alpha, gamma=1.0):
    V = defaultdict(float)
    for _ in range(n_episodes):
        S = env.reset()
        while True:
            action = policy(S)
            next_state, reward, done = env.step(action)
            V[S] = V[S] + alpha * (reward + gamma*V[next_state] - V[S])
            S = next_state
            if done:
                break
    return V

def policy(S):
    score, dealer_score, ace = S
    if score >= 20:
        return 0
    else:
        return 1

V = td_evaluation(100000, alpha=0.05)
plot.plot_blackjack(V)
