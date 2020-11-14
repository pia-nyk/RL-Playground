import gym
from collections import defaultdict
from gym import spaces
import random

env = gym.make("Blackjack-v0")
action_space = env.action_space
policy = {}
# print(score_space)

def initialize_policy():
    for a in range(32):
        for b in range(11):
            for c in [True, False]:
                tup = (a, b, c)
                policy[tup] = action_space.sample()

def sample_policy(observation):
    tup = (observation[0], observation[1], observation[2])
    return policy[tup]


def generate_episode():
    states_actions = []
    states = []
    rewards = []
    observation = env.reset()
    while True:
        action = sample_policy(observation)
        observation, reward, done, _ = env.step(action)
        # print (reward)
        states.append(observation)
        states_actions.append((observation, action))
        rewards.append(reward)
        if done:
            break
    return states_actions, rewards, states
#
def monte_carlo_ES(env, n_episodes):
    Q_table = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        states_actions, rewards, states = generate_episode()
        returns = 0
        # print(states_actions)
        for state in range(len(states_actions)):
            R = rewards[state]
            S = states_actions[state]
            returns+=R
            if S not in states_actions[:state]:
                N[S]+=1
                Q_table[S]+=(returns - Q_table[S])/N[S]
        for state in states:
            tup1 = (state, 0)
            tup2 = (state, 1)
            if Q_table[tup1] > Q_table[tup2]:
                policy[state] = 0
            else:
                policy[state] = 1
    return policy

initialize_policy()
policy = monte_carlo_ES(env, 100000)
print(policy)
