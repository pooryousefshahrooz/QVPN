#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import numpy as np
import random
from tqdm import tqdm
import multiprocessing as mp
from absl import app
from absl import flags
import ast
import tensorflow as tf
import sys

# sys.path.insert(0, '../')
from game import CFRRL_Game
from model import Model
# from network import Network
# from config import get_config
from solver import Solver
import threading
import time
import os
import shutil
FLAGS = flags.FLAGS
# flags.DEFINE_integer('num_agents',1, 'number of agents')
# flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseline')
# flags.DEFINE_integer('num_iter', 20, 'Number of iterations each agent would run')
# print(FLAGS.num_agents)
import pdb
# num_agents = 1
# num_iter =20
# baseline = "avg"
# pdb.set_trace()
GRADIENTS_CHECK=False


# In[ ]:





# In[ ]:





# In[ ]:


class RL:
    def __init__(self,config,start = 0):
        self.num_agents = 1
        self.num_iter =config.rl_batch_size
        self.epoch_numbers = 1
        self.baseline = "avg"
        self.ckpt = ''
        self.training_testing_switching_file = "training_testing_switching_file.txt"
        
        #we use lock to write on the training and testing switching file 
        self.lock = threading.Lock()
        self.value = start
        
    
    def get_testing_flag(self):
        try:
            f = open(self.training_testing_switching_file+".txt", 'r')
        except:
            f = open(self.training_testing_switching_file, 'r')
        for line in f:
            if line:
                line = line.strip()
                link = line.split('\t')
                #print(line,link)
                step,testing_flag = link
                if testing_flag=="True":
                    return step,True
                else:
                    return step,False
        f.close()
    def set_testing_flag(self,last_step,training_flag):
        print("we are going to set the flag of testing to %s with step %s "%(training_flag,last_step))
        self.lock.acquire()
        with open(self.training_testing_switching_file, "w") as file_object:
            if training_flag:
                file_object.write(str(last_step)+"\t"+str("True")+"\n")
            else:
                file_object.write(str(last_step)+"\t"+str("False")+"\n")
        file_object.close()
        self.lock.release()
        
        try:
            f = open(self.training_testing_switching_file+".txt", 'r')
        except:
            f = open(self.training_testing_switching_file, 'r')
        for line in f:
            if line:
                line = line.strip()
                link = line.split('\t')
                #print(line,link)
                step,testing_flag = link
                print("this is what we read from file ",line)
        print("we set the flag of testing to %s with step %s "%(training_flag,last_step))
    def central_agent(self,config, topology_name,game, model_weights_queues, experience_queues):
        model = Model(config,topology_name, game.state_dims, game.action_dim, game.max_moves, master=True)
        model.save_hyperparams(config)
        start_step = model.restore_ckpt()
        for step in tqdm(range(start_step, self.epoch_numbers), ncols=70, initial=start_step):
            model.ckpt.step.assign_add(1)
            model_weights = model.model.get_weights()

#             for i in range(FLAGS.num_agents):
            for i in range(self.num_agents):
                model_weights_queues[i].put(model_weights)

            if config.method == 'actor_critic':
                #assemble experiences from the agents
                s_batch = []
                a_batch = []
                r_batch = []

#                 for i in range(FLAGS.num_agents):
                for i in range(self.num_agents):
                    s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()

#                     assert len(s_batch_agent) == FLAGS.num_iter, \
                    assert len(s_batch_agent) == self.num_iter, \
                        (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                    s_batch += s_batch_agent
                    a_batch += a_batch_agent
                    r_batch += r_batch_agent

                assert len(s_batch)*game.max_moves == len(a_batch)
                #used shared RMSProp, i.e., shared g
                actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
                value_loss, entropy, actor_gradients, critic_gradients = model.actor_critic_train(np.array(s_batch), 
                                                                        actions, 
                                                                        np.array(r_batch).astype(np.float32), 
                                                                        config.entropy_weight)

                if GRADIENTS_CHECK:
                    for g in range(len(actor_gradients)):
                        assert np.any(np.isnan(actor_gradients[g])) == False, ('actor_gradients', s_batch, a_batch, r_batch, entropy)
                    for g in range(len(critic_gradients)):
                        assert np.any(np.isnan(critic_gradients[g])) == False, ('critic_gradients', s_batch, a_batch, r_batch)

                if step % config.save_step == config.save_step -1:
                    print("going to store checkpoint in training step ",step)
                    testing_flag = False
                    last_step = step
                    while(testing_flag):
                        try:
                            last_step,testing_flag = self.get_testing_flag()
                        except:
                            pass
                        if not testing_flag:
                            print("training and the flag is %s which means we can save"%(testing_flag))
                        else:
                            print("training and the flag is %s which means we have to wait!"%(testing_flag))
                            time.sleep(2)
                        
                        last_step= int(last_step)
                    model.save_ckpt(_print=True)

                    #log training information
                    actor_learning_rate = model.lr_schedule(model.actor_optimizer.iterations.numpy()).numpy()
                    avg_value_loss = np.mean(value_loss)
                    avg_reward = np.mean(r_batch)
                    avg_entropy = np.mean(entropy)

                    model.inject_summaries({
                        'learning rate': actor_learning_rate,
                        'value loss': avg_value_loss,
                        'avg reward': avg_reward,
                        'avg entropy': avg_entropy
                        }, step)
                    print('lr:%f, value loss:%f, avg reward:%f, avg entropy:%f step %s'%(actor_learning_rate, avg_value_loss, avg_reward, avg_entropy,step))
                    self.set_testing_flag(step,True)
                   
                    
            elif config.method == 'pure_policy':
                #assemble experiences from the agents
                s_batch = []
                a_batch = []
                r_batch = []
                ad_batch = []

#                 for i in range(FLAGS.num_agents):
                for i in range(self.num_agents):
                    s_batch_agent, a_batch_agent, r_batch_agent, ad_batch_agent = experience_queues[i].get()

#                     assert len(s_batch_agent) == FLAGS.num_iter, \
                    assert len(s_batch_agent) == self.num_iter, \
                        (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent), len(ad_batch_agent))

                    s_batch += s_batch_agent
                    a_batch += a_batch_agent
                    r_batch += r_batch_agent
                    ad_batch += ad_batch_agent

                assert len(s_batch)*game.max_moves == len(a_batch)
                #used shared RMSProp, i.e., shared g
                actions = np.eye(game.action_dim, dtype=np.float32)[np.array(a_batch)]
                entropy, gradients = model.policy_train(np.array(s_batch), 
                                                          actions, 
                                                          np.vstack(ad_batch).astype(np.float32), 
                                                          config.entropy_weight)

                if GRADIENTS_CHECK:
                    for g in range(len(gradients)):
                        assert np.any(np.isnan(gradients[g])) == False, (s_batch, a_batch, r_batch)

                if step % config.save_step == config.save_step -1:
                    print("2 going to store checkpoint in training step ",step)
                    testing_flag = False
                    last_step = step
                    while(testing_flag):
                        try:
                            last_step,testing_flag = self.get_testing_flag()
                        except:
                            pass
                        
                        if not testing_flag:
                            print("training and the flag is %s which means we can save"%(testing_flag))
                        else:
                            print("training and the flag is %s which means we have to wait!"%(testing_flag))
                            time.sleep(2)
                        last_step= int(last_step)
                    model.save_ckpt(_print=True)

                    #log training information
                    learning_rate = model.lr_schedule(model.optimizer.iterations.numpy()).numpy()
                    avg_reward = np.mean(r_batch)
                    avg_advantage = np.mean(ad_batch)
                    avg_entropy = np.mean(entropy)
                    model.inject_summaries({
                        'learning rate': learning_rate,
                        'avg reward': avg_reward,
                        'avg advantage': avg_advantage,
                        'avg entropy': avg_entropy
                        }, step)
                    print('lr:%f, avg reward:%f, avg advantage:%f, avg entropy:%f step %s'%(learning_rate, avg_reward, avg_advantage, avg_entropy,step))
                    self.set_testing_flag(step,True)
                
                    
    def agent(self,agent_id, config, game,network, wk_subset, model_weights_queue, experience_queue):
        random_state = np.random.RandomState(seed=agent_id)
        model = Model(config,network.topology_name, game.state_dims, game.action_dim, game.max_moves, master=False)
        solver = Solver()
        # initial synchronization of the model weights from the coordinator 
        model_weights = model_weights_queue.get()
        model.model.set_weights(model_weights)

        idx = 0
        s_batch = []
        a_batch = []
        r_batch = []
        if config.method == 'pure_policy':
            ad_batch = []
        run_iteration_idx = 0
        num_wks = len(wk_subset)
        random_state.shuffle(wk_subset)
        run_iterations = self.num_iter
        print("we are in agent ")
        while True:
            wk_idx = wk_subset[idx]
            print("training work load %s from %s "%(wk_idx,len(wk_subset)))
            network.get_each_user_all_paths(wk_idx,False)
            #state
            state = game.get_state(wk_idx,network,False)
            s_batch.append(state)
            #action
            if config.method == 'actor_critic':    
                policy = model.actor_predict(np.expand_dims(state, 0)).numpy()[0]
            elif config.method == 'pure_policy':
                policy = model.policy_predict(np.expand_dims(state, 0)).numpy()[0]
            #print("np.count_nonzero(policy) >= game.max_moves, (policy, state)",np.count_nonzero(policy) , game.max_moves)
            #print("(policy that are all paths %s, state %s)"%(policy, state))
            
            assert np.count_nonzero(policy) >= game.max_moves, (policy, state)
            actions = random_state.choice(game.action_dim, game.max_moves, p=policy, replace=False)
#             print("we selected top paths %s which are %s"%(len(actions),actions))
            
            for a in actions:
                a_batch.append(a)

            #reward
            reward = game.reward(wk_idx,network,actions,solver)
            
            print("training for workload %s got reward %s "%(wk_idx,reward))
            #print("reward is ",reward)
            r_batch.append(reward)

            if config.method == 'pure_policy':
                #advantage
                if config.baseline == 'avg':
                    ad_batch.append(game.advantage(wk_idx, reward))
                    game.update_baseline(wk_idx, reward)
                elif config.baseline == 'best':
                    best_actions = policy.argsort()[-game.max_moves:]
                    best_reward = game.reward(wk_idx, best_actions)
                    ad_batch.append(reward - best_reward)

            run_iteration_idx += 1
            if run_iteration_idx >= run_iterations:
                # Report experience to the coordinator                          
                if config.method == 'actor_critic':    
                    experience_queue.put([s_batch, a_batch, r_batch])
                elif config.method == 'pure_policy':
                    experience_queue.put([s_batch, a_batch, r_batch, ad_batch])

                #print('report', agent_id)

                # synchronize the network parameters from the coordinator
                model_weights = model_weights_queue.get()
                model.model.set_weights(model_weights)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                if config.method == 'pure_policy':
                    del ad_batch[:]
                run_iteration_idx = 0
            # Update idx
            idx += 1
            if idx == num_wks:
               random_state.shuffle(wk_subset)
               idx = 0

    def train(self,config,network):
        #cpu only
        tf.config.experimental.set_visible_devices([], 'GPU')
        self.training_testing_switching_file = config.training_testing_switching_file
        #tf.get_logger().setLevel('INFO')
        #tf.debugging.set_log_device_placement(True)

        #config = get_config(FLAGS) or FLAGS
        """we first find the candidate paths and use it for action dimention"""
        # we se the state dimention and action dimention
        game = CFRRL_Game(config,network)
        self.epoch_numbers = config.max_step
        self.set_testing_flag(1,True)
        model_weights_queues = []
        experience_queues = []
#                                             if FLAGS.num_agents == 0 or FLAGS.num_agents >= mp.cpu_count():
#                                                 FLAGS.num_agents = mp.cpu_count() - 1
#                                             print('Agent num: %d, iter num: %d\n'%(FLAGS.num_agents+1, FLAGS.num_iter))
#                                             for _ in range(FLAGS.num_agents):
        if self.num_agents == 0 or self.num_agents >= mp.cpu_count():
            self.num_agents = mp.cpu_count() - 1
        print('Agent num: %d, iter num: %d\n'%(self.num_agents+1, self.num_iter))
        for _ in range(self.num_agents):
            model_weights_queues.append(mp.Queue(1))
            experience_queues.append(mp.Queue(1))

#                                             tm_subsets = np.array_split(game.wk_indexes, FLAGS.num_agents)
        wk_subsets = np.array_split(game.wk_indexes, self.num_agents)

        coordinator = mp.Process(target=self.central_agent, args=(config,network.topology_name, game, model_weights_queues, experience_queues))

        coordinator.start()

        agents = []
#                                             for i in range(FLAGS.num_agents):
        for i in range(self.num_agents):
            agents.append(mp.Process(target=self.agent, args=(i, config, game, network,wk_subsets[i], model_weights_queues[i], experience_queues[i])))

        #for i in range(FLAGS.num_agents):
        for i in range(self.num_agents):
            agents[i].start()

        coordinator.join()

    def sim(self,config, model,network,solver, game,wk_idx):
        #for wk_idx in game.wk_indexes:
        #print("*************************** getting the state for work load %s ****************** "%(wk_idx))
        state = game.get_state(wk_idx,network,True)
        if config.method == 'actor_critic':
            policy = model.actor_predict(np.expand_dims(state, 0)).numpy()[0]
        elif config.method == 'pure_policy':
            policy = model.policy_predict(np.expand_dims(state, 0)).numpy()[0]
        actions = policy.argsort()[-game.max_moves:]
        egr = game.evaluate(wk_idx,network,solver,"RL",actions) 
        return actions,egr
    def test(self,config,network):
        self.epoch_numbers = config.max_step
        last_step = 1
        new_last_step = 1
        testing_flag = True
        solver = Solver()
        print("****************** testing ********************************")
        while(last_step<self.epoch_numbers):
            #print("going to call self.get_testing_flag(last_step) ******************** ")
            while(not testing_flag):
                try:
                    new_last_step,testing_flag = self.get_testing_flag()
                    print(" testing we read the file and it was flag %s step %s"%(testing_flag,new_last_step))
                except:
                    pass
    #             print("******* this is our step and flag %s %s **********"%(last_step,testing_flag))
                if not testing_flag:
                    print("testing and the flag is %s which means we have to wait!"%(testing_flag))
                    time.sleep(2)
                else:
                    print("testing and the flag is %s which means we can test :)"%(testing_flag))
                    
                new_last_step = int(new_last_step)
            
            if new_last_step>last_step:
                
                last_step = new_last_step
                self.ckpt = ''
                #Using cpu for testing
                tf.config.experimental.set_visible_devices([], 'GPU')
                tf.get_logger().setLevel('INFO')
                
                each_wk_scheme_egr ={}
                each_wk_idx_optimal = {}
                """we first find the candidate paths and use it for action dimention"""
                # we set the state dimention and action dimention
                game = CFRRL_Game(config, network)
                model = Model(config,network.topology_name, game.state_dims, game.action_dim, game.max_moves)
        #         last_chckpoint = model.restore_ckpt(FLAGS.ckpt)
                last_chckpoint = model.restore_ckpt(self.ckpt)
                model = Model(config,network.topology_name, game.state_dims, game.action_dim, game.max_moves)
        #         model = Model(config, game.state_dims, game.action_dim, game.max_moves)
        #             current_chckpoint = model.restore_ckpt(FLAGS.ckpt)
                current_chckpoint = model.restore_ckpt(self.ckpt)
                if config.method == 'actor_critic':
                    learning_rate = model.lr_schedule(model.actor_optimizer.iterations.numpy()).numpy()
                elif config.method == 'pure_policy':
                    learning_rate = model.lr_schedule(model.optimizer.iterations.numpy()).numpy()
                print('\nstep %d, learning rate: %f\n'% (current_chckpoint, learning_rate))
                time_in_seconds = time.time()
                for wk_idx in range(len(game.testing_wk_indexes)):
                    
                    print(" *** going to get the paths of all users in workload %s out of %s ***"%(wk_idx,len(game.testing_wk_indexes)))
                    network.get_each_user_all_paths(wk_idx,False)
                    actions,rl_egr= self.sim(config,model,network,solver,game,wk_idx)
                    print("testing for workload %s got egr %s "%(wk_idx,rl_egr))
                    network.save_results(wk_idx,config,False,True,False,False,new_last_step,rl_egr,0,0,time_in_seconds)
                    print("we saved the results in file!")
                    if config.save_rl_results_for_initialization and new_last_step in config.set_of_epoch_for_saving_rl_results_for_ga:
                        print("we have wk %s epoch number %s and target to save path information are %s"%(wk_idx,new_last_step,config.set_of_epoch_for_saving_rl_results_for_ga))
                        print("we are going to save")
                        network.save_rl_results_for_genetic_initialization(config,wk_idx,new_last_step,rl_egr)
                    else:
                        print("no save! to initialize genetic epoch number %s worklosd %s from %s "%(new_last_step,wk_idx,len(game.testing_wk_indexes)))
                    
                    if new_last_step%100==0:
                        print(" ****epoch #",new_last_step,"# paths",network.num_of_paths,"wk_idx",wk_idx,
                                    "RL",rl_egr)
                self.set_testing_flag(new_last_step,False)
                    
            else:
                print("the flag for testing is set to true %s but we already have trained for this epoch number %s "%(testing_flag,new_last_step))
                time.sleep(3)
                new_last_step,testing_flag = self.get_testing_flag()
                new_last_step = int(new_last_step)
                print(" testing 2: we read the file and it was flag %s step %s"%(testing_flag,new_last_step))

                


# In[ ]:


# rl = RL()
# rl.main()


# In[ ]:


# if __name__ == '__main__':
#     app.run(main)


# In[ ]:





# In[1]:


# import numpy as np
# tm_indexes = np.arange(0, 100)
# print(tm_indexes)


# In[ ]:




