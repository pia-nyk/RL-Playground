'''
    References: https://medium.com/@shiyan/xavier-initialization-and-batch-normalization-my-understanding-b5b91268c25c
    https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

'''

import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')
state_size = 4
action_size = env.action_space.n

#hyperparameters
n_episodes = 300
learning_rate = 0.01
gamma = 0.95

def discounted_normalized_returns(episode_rewards):
    discounted_rewards = np.zeros(episode_rewards)
    total_reward = 0
    #total rewards
    for i in reversed(range(episode_rewards)):
        total_reward = total_reward * gamma + episode_rewards[i]
        discounted_rewards[i] = total_reward
    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)
    #normalizing the rewards of the episode
    discounted_normalized_rewards = (discounted_rewards - mean)/std

with tf.name_scope("inputs"):
    inputs = tf.placeholder(tf.float32, [None, state_size], name="inputs_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions_")
    discounted_rewards = tf.placeholder(tf.float32, [None,], name="discounted_rewards")

with tf.name_scope("fc1"):
    #xavier_initialization used to avoid stauration points
    fc1 = tf.contrib.layers.fully_connected(inputs = "inputs_", num_outputs=10, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initialization())

with tf.name_scope("fc2"):
    fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs=action_size, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initialization())

with tf.name_scope("fc3"):
    fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs=action_size, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initialization())

with tf.name_scope("softmax"):
    action_distribution = tf.nn.softmax(fc3)

with tf.name_scope("loss"):
    #cross entropy of results
    neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels=action)
    #gradient ascent formula
    loss = tf.reduce_mean(neg_loss_prob * discounted_rewards)

with tf.name_scope("train"):
    #should this be maximize?
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def train():
    allRewards = []
    total_rewards = 0
    maximumRewardRecorded = 0
    episode = 0

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #initialize tf variables defined
        sess.run(tf.global_variables_initializer())

        #train for # n_episodes
        for _ in range(n_episodes):
            states, actions, rewards = [],[],[]
            episode_rewards_sum = 0
            observation = env.reset()
            done = False

            #horizon = episode length
            while not done:
                #bring the oservation in required shape and pass to action_distribution func to get prob distributions
                action_prob_distribution = sess.run(action_distribution, feed_dict={input_:observation.reshape([1, 4])})
                #choose an action from the acquired probability distribution
                action = np.random.choice(action_space, p=action_prob_distribution.ravel())
                new_observation, reward, done, _ = env.step(action)

                #store the episode info
                states.append(observation)
                actions_ = np.zeros(action_size)
                actions_[action] = 1
                actions.append(action)
                rewards.append(reward)
                observation = new_observation

            #at the end of the episode
            episode_rewards_sum = np.sum(rewards)
            allRewards.append(episode_rewards_sum)
            total_rewards = np.sum(allRewards)
            mean_reward = np.mean(total_rewards)
            maximumRewardRecorded = np.amax(allRewards)

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", episode_rewards_sum)
            print("Mean Reward", mean_reward)
            print("Max reward so far: ", maximumRewardRecorded)

            #get discounted reward for training
            discounted_normalized_rewards = discounted_normalized_returns(episode_rewards)
            loss, _ = sess.run([loss, train_opt], feed_dict={inputs_:np.vstack(np.array(states)), actions_:np.vstack(np.array(actions)), discounted_rewards:discounted_normalized_rewards})

            if _%100 == 0:
                saver.save(sess, "./models/model.ckpt")
                print("Model saved")

def evaluate():
    with tf.Session as sess:
        rewards = []
        saver.restore(sess, "./model/model.ckpt")
        for _ in range(10):
            observation = env.reset()
            step = 0
            while True:
                action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,4])})
                action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())
                new_state, reward, done, info = env.step(action)
                total_rewards += reward
                if done:
                    rewards.append(total_rewards)
                    print ("Score", total_rewards)
                    break
                observation = new_state
        env.close()
        print ("Score over time: " +  str(sum(rewards)/10))
