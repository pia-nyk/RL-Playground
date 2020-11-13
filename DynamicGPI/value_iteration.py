import gym
import numpy as np

#create environment
env = gym.make('FrozenLake-v0')
#as the environment is continuous, there cannot be finite number of states
states = env.observation_space.n


#check the number of actions
actions = env.action_space.n

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

def value_iteration(threshold=0.001, gamma=1.0):
    #one step lookahead from the state to get the greedy action
    def one_step_lookahead(state, V):
        A = np.zeros(actions)
        for a in range(actions):
            for prob, next_state, reward, done in env.env.P[state][a]:
                A[a]+=prob*(reward+gamma*V[next_state])
        return A

    V = np.zeros(states)
    while True:
        delta = 0  #checker for improvements across states
        for state in range(env.env.nS):
            act_values = one_step_lookahead(state,V)  #lookahead one step
            best_act_value = np.max(act_values) #get best action value
            delta = max(delta,np.abs(best_act_value - V[state]))  #find max delta across all states
            V[state] = best_act_value  #update value to best action value
        if delta < threshold:  #if max improvement less than threshold
            break
    policy = np.zeros([env.env.nS, env.env.nA])
    for state in range(env.env.nS):  #for all states, create deterministic policy
        act_val = one_step_lookahead(state,V)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1
    return policy, V

val_iter_policy = value_iteration()
final_policy = val_iter_policy[0]
print(final_policy)
view_policy(final_policy)
