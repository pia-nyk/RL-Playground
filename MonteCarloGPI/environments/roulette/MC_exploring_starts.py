import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make("Roulette-v0")
EPS = 0.05
GAMMA = 1.0

Q = {}
G = 0

state_space = []
returns = {}
pairs_visited = {}

roulette_states = 37
roulette_actions = 37

for state in range(roulette_states):
    for action in range(roulette_actions):
        Q[(state, action)] = 0
        returns[(state, action)] = 0
        pairs_visited[(state, action)] = 0

#initialize random policy
policy = {}
for state in range(roulette_states):
    policy[state] = np.random.choice(roulette_actions)

num_episodes = 100000
for i in range(num_episodes):
    stateActionReturns = []
    memory = []
    if i%10000 == 0:
        print("starting episode", i)
        print(policy)
    observation = env.reset()
    done = False

    while not done:
        action = policy[observation]
        observation_new, reward, done, info = env.step(action)
        memory.append((observation, action, reward))
        observation = observation_new
    memory.append((observation, action, reward))

    last = True
    for observed, action, reward in reversed(memory):
        stateActionReturns.append((observed,action,G))
        G = GAMMA*G + reward
    stateActionReturns.reverse()
    statesActionVisited = []

    for observed, action, G in stateActionReturns:
        sa = (observed, action)
        if sa not in statesActionVisited:
            pairs_visited[sa] +=1
            returns[(sa)] += (1 / pairs_visited[(sa)])*(G-returns[(sa)])
            Q[sa] = returns[sa]
            rand = np.random.random()
            if rand < 1 - EPS:
                state = observed
                values = np.array([Q[(state, a)] for a in range(roulette_actions) ])
                best=np.random.choice(np.where(values==values.max())[0])
                policy[state] = best
            else:
                policy[state] = np.random.choice(37)
            statesActionVisited.append(sa)
    if EPS - 1e-7 >0:
        EPS -= 1e-7
    else:
        EPS = 0

#testing
numEpisodes = 1000
rewards = np.zeros(numEpisodes)
totalReward = 0
wins = 0
losses = 0
print('getting ready to test policy')
for i in range(numEpisodes):
    observation = env.reset()
    done = False
    while not done:
        action = policy[observation]
        observation_, reward, done, info = env.step(action)
        observation = observation_
    totalReward += reward
    rewards[i] = totalReward
    if reward >= 1:
        wins += 1
    elif reward == -1:
        losses += 1

wins /= numEpisodes
losses /= numEpisodes
print('win rate', wins, 'loss rate', losses)
plt.plot(rewards)
plt.show()
