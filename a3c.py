'''
    Reference blog: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/A3C/
    https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
'''
import tensorflow as tf
import multiprocessing
import os
import threading

lr_a = 0.001 #actors learning rate
lr_c = 0.001 #critics learning rate
GLOBAL = True
UPDATE_GLOBAL_NET = 10
n_worker = multiprocessing.cpu_count() #no. of workers depend on the number of cpus available
log_dir = './log'
worker_episodes = 100
GAMMA = 0.75


class ActorNet():
    pass

class CriticNet():
    pass

class Worker():
    def __init__(self, name, ACTOR, CRITIC):
        self.env = gym.make('CartPole-v0')
        self.name = name
        self.ACTOR_NET = ActorNet(name, ACTOR)
        self.CRITIC_NET = CriticNet(name, CRITIC)

    def work(self):
        global _ = 0
        step_count = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        #each thread will return should_stop = True when coordinator.request_stop() is called
        #run the episode for a specified number of times
        while not THREAD_COORDINATOR.should_stop() and _ < worker_episodes:
            observation = self.env.reset()
            episode_rewards = 0 #initialize rewards gained from this episode
            while True:
                action = self.ACTOR_NET.choose_action(observation) #actor chooses action
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
                        _+=1
                    else:
                        #??
                        v_s_ = SESS.run(self.CRITIC_NET.v, {self.ACTOR_NET.s:new_obs[np.newaxis, :]})[0,0]
                    #after each episode or/and 10 steps
                    #get the calculated values from episode rewards to update the global net
                    buffer_v_target = []
                    #calculating rewards from backwards
                    for r in buffer_r[::-1]:
                        v_s_ = r  + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    #stacking array vertically
                    #todo: try removing this, its redundant, has no effect
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict_a = {
                        self.ACTOR_NET.s = buffer_s,
                        self.ACTOR_NET.a_history = buffer_a,
                        self.ACTOR_NET.v_target = buffer_v_target
                    }
                    self.ACTOR_NET.update_global(feed_dict)

                    feed_dict_c = {
                        self.CRITIC_NET.s = buffer_s,
                        self.CRITIC_NET.v_target = buffer_v_target
                    }
                    self.CRITIC_NET.update_global(feed_dict)
                if done:
                    #todo: record all running episode rewards for plot
                    pass




if __name__ == '__main__':
    SESS = tf.Session()
    with tf.device('/cpu:0'):
        #actor net optimizer function
        OPT_A = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
        #critics net optimizer function
        OPT_C = tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')
        #main actor and critic net
        ACTOR = ActorNet(GLOBAL)
        CRITIC = CriticNet(GLOBAL)
        workers = []
        for i in range(n_worker):
            worker_name = 'W_%i' % i #worker name
            workers.append(Worker(worker_name, ACTOR, CRITIC)) #every worker is an object of class Worker

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
            t = threading.thread(worker.work())
            worker_threads.append(t)
            t.start()
        THREAD_COORDINATOR.join(worker_threads) #pass on the threads to the coordinator
