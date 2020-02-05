import gym
import numpy as np
from statistics import mean

env = gym.make('FrozenLake-v0')
env.reset()

#number of states
states = env.observation_space.n
#number of actions
actions = env.action_space.n

LARGE_NUMBER = 10

def select_random_action_for_each_state():
    random_policy = np.random.choice(actions, states)
    for state in random_policy:
        state = np.random.choice(actions)
    return random_policy

def evaluate_policy_with_exploring_starts(policy, gamma):
    s = env.observation_space.sample()
    a = env.action_space.sample()
    sar_list = []

    while True:
        next_state, reward, finished, _ = env.step(a)
        env.render()
        if finished:
            break
        a = policy[next_state]
        sar_list.append((next_state,a,reward))
        s = next_state

    #compute the rewards
    G = 0
    #init empty list of states, actions and returns
    sag_list = []

    #loop from end to account for discounted returns
    # print(len(sar_list))
    for a in reversed(sar_list):
        G = a[2] + gamma*G
        sag_list.append((a[0],a[1],G))
        print (sag_list)

    #return the computed list
    return reversed(sag_list)



def improve_policy_wth_exploring_start():
    policy = select_random_action_for_each_state()
    print ("Policy: ")
    print(policy)
    Q = np.zeros([states, actions])
    rewards = np.zeros([states, actions])
    num_rewards = np.zeros([states, actions])

    i=0
    while i<LARGE_NUMBER:
        sag_list = evaluate_policy_with_exploring_starts(policy, gamma=0.9)
        visited_state_actions = []

        for a in sag_list:
            if (a[0],a[1]) not in visited_state_actions:
                rewards[a[0]][a[1]]+=a[2]
                num_rewards[a[0]][a[1]]+=1
                Q[a[0]][a[1]] = rewards[a[0]][a[1]]/num_rewards[a[0]][a[1]]
                visited_state_actions.append((a[0],a[1]))
            # print(a)
        print("Q here: ")
        print(Q)

        #update policy
        for state in range(states):
            policy[state] = np.argmax(Q[state])
        #
        print ("Policy after iteration %d : %s" %(i, policy))
        i+=1



if __name__ == '__main__':
    improve_policy_wth_exploring_start()
