
import gym
import numpy as np
#make environment
env = gym.make('FrozenLake-v0')

#as the environment is continuous, there cannot be finite number of states
states = env.observation_space.n

#check the number of actions
actions = env.action_space.n

def policy_evaluation(policy, threshold=0.0001, gamma=1.0):
    V = np.zeros(states)
    while True:
        delta = 0
        for state in range(states):
            val = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.env.P[state][action]:
                    val += action_prob * prob * (reward + gamma*V[next_state])
            delta = max(delta, np.abs(val - V[state]))
            V[state] = val
        if delta < threshold:
            break
    return np.array(V)

def policy_iteration(gamma=1.0):
    #one step lookahead from the state to get the greedy action
    def one_step_lookahead(state, V):
        A = np.zeros(actions)
        for a in range(actions):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a]+=prob*(reward+gamma*V[next_state])
        return A
    policy = np.ones([states, actions])/actions

    while True:
        current_policy = policy_evaluation(policy)
        policy_stable = True
        for state in range(states):
            chosen_action = np.argmax(policy[state])
            action_values = one_step_lookahead(state, current_policy)
            best_action = np.argmax(action_values)
            if chosen_action != best_action:
                policy_stable = False
            policy[state] = np.zeros(actions)
            policy[state][best_action] = 1
        if policy_stable:
            return policy, current_policy
    return policy, np.zeros(states)

def view_policy(policy):
    curr_state = env.reset()
    counter = 0
    reward = None
    done = False
    while not done:
        state, reward, done, _ = env.step(np.argmax(policy[curr_state]))
        curr_state = state
        counter += 1
        env.env.s = curr_state
        env.render()

pol_iter_policy = policy_iteration()
final_policy = pol_iter_policy[0]
print(final_policy)
view_policy(final_policy)
