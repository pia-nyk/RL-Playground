import gym
import numpy as np

#create environment
env = gym.make('FrozenLake-v0')
#as the environment is continuous, there cannot be finite number of states
states = env.observation_space.n


#check the number of actions
actions = env.action_space.n

#initialize value table randomly
value_table = np.zeros((states, 1))

def value_iteration(env, n_iterations, gamma=1.0, threshold=1e-30):
    for i in range(n_iterations):
        new_value_table = np.copy(value_table)
        for state in range(states):
            q_value = []
            for action in range(actions):
                next_state_reward = []
                for next_state_parameter in env.env.P[state][action]:
                    transition_prob, next_state, reward_prob, _ = next_state_parameter
                    reward = transition_prob*(reward_prob + gamma*new_value_table[next_state])
                    next_state_reward.append(reward)
                q_value.append(np.sum(next_state_reward))
            value_table[state] = max(q_value)

        #np.fabs is used to get the absolute value element wise
        if np.sum(np.fabs(new_value_table - value_table) <= threshold):
            break
    return value_table

def policy_extraction(value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += transition_prob* (reward_prob + gamma*value_table[next_state])
        policy[state] = np.argmax(Q_table)
    return policy

value_table = value_iteration(env, 1000)
print (value_table)
policy = policy_extraction(value_table)
print(policy)
