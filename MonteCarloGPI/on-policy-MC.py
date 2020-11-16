import gym
import numpy as np
import plot
from collections import defaultdict

env = gym.make("Blackjack-v0")
if not hasattr(env, 'nb_actions'): env.nb_actions = 2
if not hasattr(env, 'act_space'): env.act_space = [0, 1]

def sample_policy(S, prob):
    return np.random.choice(env.act_space, p=[prob[(S, a)] for a in env.act_space])

def argmax(arr):
    return np.random.choice(np.flatnonzero(arr == np.max(arr)))

def generate_episode(prob):
    sa_list, rewards = [],[]
    observation = env.reset()
    while True:
        action = sample_policy(observation, prob)
        sa_list.append((observation,action))

        observation, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            break
    return sa_list, rewards

def on_policy_MC_control(n_episodes, eps, gamma=1.0):
    prob = defaultdict(lambda: 1/env.nb_actions)
    Q = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(n_episodes):
        sa_list, rewards = generate_episode(prob)
        G = 0
        for s in range(len(sa_list)-2,-1,-1):
            state = sa_list[s][0]
            action = sa_list[s][1]
            reward = rewards[s+1]
            G = G*gamma + reward
            if not (state,action) in sa_list[:s]:
                returns[(state,action)].append(G)
                Q[(state,action)] = np.average(returns[(state, action)])
            Astar = argmax([Q[(state,action)] for a in range(env.nb_actions)])
            for a in range(env.nb_actions):
                if a == Astar:
                    prob[(state,a)] = 1 - eps + eps/env.nb_actions
                else:
                    prob[(state,a)] = eps/env.nb_actions
    return Q, prob

Q, prob = on_policy_MC_control(n_episodes=1000000, eps=0.1)
plot.plot_blackjack(Q)
