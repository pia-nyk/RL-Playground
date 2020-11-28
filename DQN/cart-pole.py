'''
    Deep Q-Learning on OpenAI Gym Cartpole env
    Reference: https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
'''

import gym
import tensorflow as tf
import numpy as np
from gym import wrappers
import os
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

class DQNModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DQNModel, self).__init__() #calling the parent class constructor
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,)) #input layer takes the observations
        self.hidden_layers = []
        for i in hidden_units:
            #hidden layer activation kept as tanh for better convergence purposes, initailizing the layers with random values from
            #Normal distribution
            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='tanh', kernel_initializer='RandomNormal'))
        #output layer activation - linear, as q-value can take on any real value
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function #for a callable tensorflow graph
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        outputs = self.output_layer(z)
        return outputs


class DQN():
    def __init__(self, num_states, num_actions, hidden_states, gamma, max_experiences, min_experiences, batch_size, learning_rate):
        self.num_actions = num_actions
        self.gamma = gamma #dicount factor
        self.batch_size = batch_size #batch size to get training data for experience replay buffer
        self.optimizer = optimizers.Adam(learning_rate)
        self.model = DQNModel(num_states, hidden_states, num_actions)
        self.experience = {'s':[], 'a':[], 'r':[], 's2':[],'done':[]} #experience replay buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32'))) #want the input to be of atleast 2d

    #epsilon greedy policy
    def policy(self, states, eps):
        if np.random.random() > eps:
            return np.argmax(self.predict(np.atleast_2d(states))[0])
        else:
            return np.random.choice(self.num_actions)

    #target net with stable q values for reference is passed for training the actual net
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences: #start training once the min number of data is available in exp replay buffer
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size) #get batchsize exp data for training
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1) #choose the q-value with max approximation
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next) #if done then reward else value update
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    #adding the state, action, reward, next state values in experience buffer
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0) #remove an old experience to make place for a new one FIFO
        for key, value in exp.items():
            self.experience[key].append(value) #add the new experience

    #after a specified number of training steps, copy the trained value in local net to target net for future ref
    def copy_weights(self, TargetNet):
        local_net_weights = self.model.trainable_variables
        target_net_weights = TargetNet.model.trainable_variables
        #manually set the trainable variables of TrainNet to TargetNet
        for v1, v2 in zip(local_net_weights, target_net_weights):
            v1.assign(v2.numpy())

def play_game(env, TrainNet, TargetNet, eps, copy_steps):
    #initialize params
    rewards = 0
    iter = 0
    done = False
    observation = env.reset()
    losses = []

    while not done:
        action = TrainNet.policy(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)
        rewards+=reward
        if done:
            reward = -200 #negative in case the pole falls and episode ends
            env.reset()
        #store the exp in buffer
        experience = {'s': prev_observation, 'a': action, 'r': rewards, 's2': observation, 'done': done}
        TrainNet.add_experience(experience)
        loss = TrainNet.train(TargetNet)
        # print("loss : " + str(loss))
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter+=1
        if iter % copy_steps == 0: #copying the weights to TargetNet at specified intervals
            TargetNet.copy_weights(TrainNet)
    return rewards, np.mean(losses)

def make_video(env, TrainNet):
    #run the cartpole env for one episode, gym will create a video of it
    # env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        action = TrainNet.policy(observation, 0)
        env.render()
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))

def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_step = 25
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 500
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay) #decaying epsilon value
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("avg reward for last 100 episodes:", avg_rewards)
    fig = plt.figure()
    plt.plot(np.arange(1, len(total_rewards)+1), total_rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    env.close()

if __name__ == '__main__':
    main()
