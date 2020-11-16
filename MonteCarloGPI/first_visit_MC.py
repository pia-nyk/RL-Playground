import gym
from collections import defaultdict
import plot
import numpy as np

env = gym.make("Blackjack-v0")

def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score>=20 else 1

def generate_episode(policy):
    states, actions, rewards = [],[],[]
    observation = env.reset()
    while True:
        states.append(observation)
        action = sample_policy(observation)
        actions.append(action)

        observation, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            break
    return states, actions, rewards

def first_visit_MC(policy, env, n_episodes, gamma=1.0):
    V = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(n_episodes):
        states, _, rewards = generate_episode(policy)
        G = 0

        for state in range(len(states)-2,-1,-1):
            R1 = rewards[state+1]
            S = states[state]
            G = G*gamma + R1

            if S not in states[:state]:
                returns[S].append(G)
                V[S] = np.average(returns[S])
    return V

V = first_visit_MC(sample_policy, env, 100000)
plot.plot_blackjack(V)
