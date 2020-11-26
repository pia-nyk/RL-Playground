import gym
import tensorflow as tf
import numpy as np

class DQNModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(Model, self).__init__() #calling the parent class constructor
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
        self.optimizer = tf.optimizer.Adam(learning_rate)
        self.model = DQNModel(num_states, hidden_states, num_actions)
        self.experience = {'s':[], 'a':[], 'r':[], 'done':[]} #experience replay buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32'))) #want the input to be of atleast 2d

    #epsilon greedy policy
    def policy(self, states, eps):
        if np.rand.random() > eps:
            return np.argmax(self.predict(states)[0])
        else:
            return np.random.choice(self.num_actions)

    #target net with stable q values for reference is passed for training the actual net
    def train(self, TargetNet):
        pass

    #adding the state, action, reward, next state values in experience buffer
    def add_experience(self, exp):
        if len(self.experience['s']) >= max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0) #remove an old experience to make place for a new one FIFO
            for key, value in exp.items():
                self.experience[key].append(value) #add the new experience

    #after a specified number of training steps, copy the trained value in local net to target net for future ref
    def copy_weights(self, TargetNet):
        pass
