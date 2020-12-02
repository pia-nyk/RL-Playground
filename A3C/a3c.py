'''
    Reference blog: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/A3C/
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import multiprocessing
import os
import threading
import numpy as np
import gym
import shutil
import matplotlib.pyplot as plt

lr_a = 0.001 #actors learning rate
lr_c = 0.001 #critics learning rate
GLOBAL = True
UPDATE_GLOBAL_NET = 10
n_worker = multiprocessing.cpu_count() #no. of workers depend on the number of cpus available
log_dir = './log'
worker_episodes = 100
GAMMA = 0.75
GLOBAL_NETWORK = 'Global_Network'
episode_count = 0
GLOBAL_RUNNING_R = []
ENTROPY_BETA = 0.001

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet():
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NETWORK:
            #initialize the variables
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                #build the actor net
                self.a_params, self.c_params = self.build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                #workers network params
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_history = tf.placeholder(tf.int32, [None,], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.a_prob, self.v, self.a_params, self.c_params = self.build_net(scope)

                #use td error for local network update
                #difference between value predicted by critic net and value obtained by
                #following actors action choices
                td = tf.subtract(self.v_target, self.v, name='TD_Error')
                with tf.name_scope('c_loss'):
                    #mean of squares of all td losses -
                    #the value with which critic net could improve calculated after every episode
                    self.c_loss = tf.reduce_mean(tf.square(td))
                    #calculating policy loss
                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_history, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    #gradients of actor net params wrt policy loss
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    #gradients of critic net params wrt value loss
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            #flow of params from local to global net and vice versa
            with tf.name_scope('sync'):
                #get the global params values
                with tf.name_scope('pull'):
                    self.pull_a_params = [local_param.assign(global_param) for local_param, global_param in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params = [local_param.assign(global_param) for local_param, global_param in zip(self.c_params, globalAC.c_params)]

                #update the global network after every episode or UPDATE_GLOBAL_NET steps, whichever is less
                with tf.name_scope('push'):
                    self.update_a_params = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_params = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def build_net(self, scope):
        #generate initial weights randomly from normal distribution, mean of 0, std of 1
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_params, self.update_c_params], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params, self.pull_c_params])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker():
    def __init__(self, name, globalAC):
        self.env = gym.make('CartPole-v0')
        self.name = name
        self.ACNet = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R
        global episode_count
        step_count = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        #each thread will return should_stop = True when coordinator.request_stop() is called
        #run the episode for a specified number of times
        while not THREAD_COORDINATOR.should_stop() and episode_count < worker_episodes:
            observation = self.env.reset()
            episode_rewards = 0 #initialize rewards gained from this episode
            while True:
                action = self.ACNet.choose_action(observation) #actor chooses action
                new_obs, reward, done, _ = self.env.step(action)
                if done:
                    reward = -5 #the pole fell down
                episode_rewards+=reward
                #store the params of this step to update main net later
                buffer_s.append(observation)
                buffer_a.append(action)
                buffer_r.append(reward)

                #update the actor critic nets every 10 steps
                if step_count % UPDATE_GLOBAL_NET == 0 or done:
                    if done:
                        #reset the variable holding v value for current episode
                        v_global_iter = 0
                        episode_count+=1
                    else:
                        #??
                        v_global_iter = SESS.run(self.ACNet.v, {self.ACNet.s:new_obs[np.newaxis, :]})[0,0]
                    #after each episode or/and 10 steps
                    #get the calculated values from episode rewards to update the global net
                    buffer_v_target = []
                    #calculating rewards from backwards
                    for r in buffer_r[::-1]:
                        v_global_iter = r  + GAMMA * v_global_iter
                        buffer_v_target.append(v_global_iter)
                    buffer_v_target.reverse()
                    #stacking array vertically
                    #todo: try removing this, its redundant, has no effect
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.ACNet.s: buffer_s,
                        self.ACNet.a_history: buffer_a,
                        self.ACNet.v_target: buffer_v_target
                    }
                    self.ACNet.update_global(feed_dict)

                    #reset the buffers
                    buffer_s, buffer_r, buffer_a = [],[],[]
                    self.ACNet.pull_global()

                observation = new_obs
                step_count+=1
                if done:
                    #todo: record all running episode rewards for plot
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break


if __name__ == '__main__':
    SESS = tf.Session()
    with tf.device('/cpu:0'):
        #actor net optimizer function
        OPT_A = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
        #critics net optimizer function
        OPT_C = tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')
        #main actor and critic net - we need only the params
        ActorCriticNet = ACNet(GLOBAL_NETWORK)
        workers = []
        for i in range(n_worker):
            worker_name = 'W_%i' % i #worker name
            workers.append(Worker(worker_name, ActorCriticNet)) #every worker is an object of class Worker

    THREAD_COORDINATOR = tf.train.Coordinator()
    #initialize the tensorrflow global variables
    SESS.run(tf.global_variables_initializer())
    if os.path.exists(log_dir):
        #deletes the directory
        shutil.rmtree(log_dir)
    #create event files in the directory and writes summaries of tf graph
    tf.summary.FileWriter(log_dir, SESS.graph)

    #create threads for each worker
    worker_threads = []
    for worker in workers:
        t = threading.Thread(worker.work())
        worker_threads.append(t)
        t.start()
    THREAD_COORDINATOR.join(worker_threads) #pass on the threads to the coordinator

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
