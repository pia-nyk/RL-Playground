import gym
import numpy as np
#make environment
env = gym.make('FrozenLake-v0')

#as the environment is continuous, there cannot be finite number of states
states = env.observation_space.n

#check the number of actions
actions = env.action_space.n

#here we know the policy
def compute_value(env, policy, gamma=1.0, threshold=1e-20):
    #initialize value table randomly
    value_table = np.zeros((states, 1))
    while True:
        new_table_value = np.copy(value_table)
        for state in range(states):
            action = int(policy[state])
            for next_state_parameters in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_state_parameters
                value_table[state] = transition_prob*(reward_prob+gamma*new_table_value[next_state])
        if np.sum(np.fabs(new_table_value - value_table)) <= threshold:
            break
    return value_table

def extract_policy(value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (transition_prob*(reward_prob+gamma*value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy

iterations = 20000
random_policy = np.zeros((states, 1))
for i in range(iterations):
    new_table_value = compute_value(env, random_policy)
    new_policy = extract_policy(new_table_value)
    random_policy = new_policy
    print (random_policy)
print (random_policy)
