import gym
import numpy as np
import time, pickle, os



def init_values(s, a, type="random"):
    if type == "ones":
        return np.ones((s,a))
    elif type == "random":
        return np.random.random((s,a))
    elif type == "zeros":
        return np.zeros((s,a))

def epsilon_greedy_actions(Q, epsilon, n_actions, s, train=False):
    if train or np.random.rand() < epsilon:
        return np.argmax(Q[s, :])
    else:
        return np.random.randint(0, n_actions)

def qlearning(alpha, gamma, epsilon, episodes, max_steps, n_tests, render=False, test=False):
    env = gym.make('Taxi-v3')
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_values(n_states, n_actions, type="ones")
    timestep_reward = []

    for episode in range(episodes):
        print(f"Episode: {episode}")
        s = env.reset()
        a = epsilon_greedy_actions(Q, epsilon, n_actions, s)
        t = 0
        total_rewards = 0
        done = False

        while t < max_steps:
            if render:
                env.render()
            t+=1
            s_, reward, done, info = env.step(a)
            total_rewards+=reward
            a_ = np.argmax(Q[s_, :])
            if done:
                Q[s,a] += alpha * (reward - Q[s,a])
            else:
                Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_]) - Q[s, a] )
            s,a = s_,a_
            if done:
                if render:
                    print(f"This episode took {t} timesteps and reward: {total_rewards}")
                timestep_reward.append(total_rewards)
                break
    if render:
        print(f"Here are the Q values:\n{Q}\nTesting now:")
    if test:
        test_agent(Q, env, n_tests, n_actions)
    return timestep_reward

def test_agent(Q, env, n_tests, n_actions, delay=1):
    for test in range(n_tests):
        print(f"Test #{test}")
        s = env.reset()
        done = False
        epsilon = 0
        while True:
            time.sleep(delay)
            env.render()
            a = epsilon_greedy_actions(Q, epsilon, n_actions, s, train=True)
            print(f"Chose action {a} for state {s}")
            s, reward, done, info = env.step(a)
            if done:
                if reward > 0:
                    print("Reached goal!")
                else:
                    print("Shit! dead x_x")
                time.sleep(3)
                break
if __name__ =="__main__":
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 10000
    max_steps = 2500
    n_tests = 2
    timestep_reward = qlearning(alpha, gamma, epsilon, episodes, max_steps, n_tests, test = True)
