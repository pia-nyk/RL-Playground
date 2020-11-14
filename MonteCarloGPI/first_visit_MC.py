import gym
from collections import defaultdict

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

def first_visit_MC(policy, env, n_episodes):
    value_table = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        states, _, rewards = generate_episode(policy)
        returns = 0

        for state in range(len(states)-1,-1,-1):
            R = rewards[state]
            S = states[state]
            returns+=R

            if S not in states[:state]:
                N[S]+=1
                value_table[S]+=(returns - value_table[S])/N[S]
    return value_table

final_dict = first_visit_MC(sample_policy, env, 500000)
for i in range(10):
  print(final_dict.popitem())
