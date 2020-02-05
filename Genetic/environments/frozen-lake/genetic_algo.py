import gym
import numpy as np
import random
import time
from gym import wrappers

def generate_random_policy():
    return np.random.choice(4, size=((16)))

def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]
    return new_policy

def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes

def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)
    return new_policy

def run_episode(env, policy, episode_len=100):
    total_reward = 0
    obs = env.reset()
    for t in range(episode_len):
        action = policy[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == '__main__':
    random.seed(1234) #so that we can recreate the results
    np.random.seed(1234)
    env = gym.make('FrozenLake-v0')
    env.seed(0)

    #Policy search
    n_policy = 1000 #no. of random policies we create
    n_steps = 20 #no. of steps to perform the algo
    start = time.time()
    policy_population = [generate_random_policy() for _ in range(n_policy)]
    for idx in range(n_steps):
        policy_scores = [evaluate_policy(env, policy) for policy in policy_population]
        print("Generate %d max score %0.2f" %(idx+1, max(policy_scores)))
        policy_ranks = list(reversed(np.argsort(policy_scores)))
        elite_set = [policy_population[x] for x in policy_ranks[:5]]#get the top 5 policies on the basis of score
        select_probs = np.array(policy_scores) / np.sum(policy_scores)
        child_set = [crossover(
            policy_population[np.random.choice(range(n_policy), p=select_probs)],
            policy_population[np.random.choice(range(n_policy), p=select_probs)])
            for _ in range(n_policy - 5)]
        mutated_list = [mutation(p) for p in child_set]
        policy_population = elite_set
        policy_population += mutated_list
    policy_score = [evaluate_policy(env, policy) for policy in policy_population]
    best_policy = policy_population[np.argmax(policy_score)]

    end = time.time()
    print("Best policy score = %0.2f."%(best_policy))
    print(" Time taken %4.4f" %(end-start))
