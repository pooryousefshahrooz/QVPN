#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import csv
from tqdm import tqdm
import numpy as np
import time
import pdb
import itertools
from itertools import combinations
# from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK

OBJ_EPSILON = 1e-12


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

class Game(object):
    def __init__(self, config, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)
        self.model_type = config.model_type



# In[ ]:


class CFRRL_Game():
    def __init__(self, config,network):
        random_seed=1000
        self.random_state = np.random.RandomState(seed=random_seed)
        self.model_type = config.model_type
        self.project_name = config.project_name
        self.each_wk_action_reward ={}
        self.each_testing_wk_action_reward = {}
        self.all_flows_across_workloads = []
        #env.num_pairs = self.num_pairs
        self.compute_state_action_dimensions(network)
        
        self.max_moves = network.num_of_organizations*network.number_of_flows*network.num_of_paths
        
        #print("self.max_moves %s self.action_dim %s self.max_moves %s self.action_dim %s"
       #       %(self.max_moves , self.action_dim, self.max_moves, self.action_dim))
        #assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        
        if config.method == 'pure_policy':
            self.baseline = {}
        print('Input dims :', self.state_dims)
        print('Output dims :', self.state_dims)
        print('Max moves :', self.max_moves)
    def compute_state_action_dimensions(self,network):
        """This function reads the workload from the topology+WK  and topology+WK2 files and set the dimensions of state and action"""
        self.all_flows_across_workloads = []
        for wk,k_flows in network.each_wk_each_k_user_pair_ids.items():
            for k,flows in k_flows.items():
                for flow in flows:
                    if flow not in self.all_flows_across_workloads:
                        self.all_flows_across_workloads.append(flow)
        
        state = np.zeros((1, len(self.all_flows_across_workloads),1), dtype=np.float32)   # state  []
        self.state_dims =  state.shape
        self.wk_indexes = np.arange(0, len(network.work_loads))
#         self.testing_wk_indexes = np.arange(0, len(network.testing_work_loads))
        self.testing_wk_indexes = np.arange(0, len(network.work_loads))
        self.action_dim = network.path_counter_id
        
    def get_state(self, wk_idx,network,testing_falg):
        state = np.zeros((1, len(self.all_flows_across_workloads),1), dtype=np.float32)   # state  []
        indx= 0
        self.all_flows_across_workloads.sort()
        for flow_id in self.all_flows_across_workloads:
            flag = False
            weight = 0
            list_of_organizations = []
            for k in network.each_wk_each_k_user_pair_ids[wk_idx]:
                list_of_organizations.append(k)
            list_of_organizations.sort()
            for k in list_of_organizations:
                #print("training for k ",k)
                list_of_user_pairs = []
                #print("network.each_wk_each_k_user_pair_ids",network.each_wk_each_k_user_pair_ids)
                for user_pair_id in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
                    list_of_user_pairs.append(user_pair_id)
                list_of_user_pairs.sort()
                if flow_id in list_of_user_pairs:
                    weight = network.each_wk_k_u_weight[wk_idx][k][flow_id]
            state[0][indx] = weight
            indx+=1
        return state
    def compute_egr(self,actions,wk_idx,network,solver):
#         network.each_wk_each_k_each_user_pair_id_paths = {}
        each_wk_k_u_path_ids = {}
        for k,user_pair_ids in network.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair in user_pair_ids:
                having_at_least_one_path_flag = False
                path_ids = network.each_user_pair_id_all_paths[user_pair]
                for path_id in path_ids:
                    if path_id in actions:
                        having_at_least_one_path_flag = True
                        try:
                            if len(each_wk_k_u_path_ids[wk_idx][k][user_pair])<network.num_of_paths:
                                try:
                                    each_wk_k_u_path_ids[wk_idx][k][user_pair].append(path_id)
                                except:
                                    try:
                                        each_wk_k_u_path_ids[wk_idx][k][user_pair] = [path_id]
                                    except:
                                        try:
                                            each_wk_k_u_path_ids[wk_idx][k]={}
                                            each_wk_k_u_path_ids[wk_idx][k][user_pair] = [path_id]
                                        except:
                                            each_wk_k_u_path_ids[wk_idx]={}
                                            each_wk_k_u_path_ids[wk_idx][k]={}
                                            each_wk_k_u_path_ids[wk_idx][k][user_pair] = [path_id]
                        except:
                            try:
                                each_wk_k_u_path_ids[wk_idx][k][user_pair] = [path_id]
                            except:
                                try:
                                    each_wk_k_u_path_ids[wk_idx][k]={}
                                    each_wk_k_u_path_ids[wk_idx][k][user_pair] = [path_id]
                                except:
                                    each_wk_k_u_path_ids[wk_idx]={}
                                    each_wk_k_u_path_ids[wk_idx][k]={}
                                    each_wk_k_u_path_ids[wk_idx][k][user_pair] = [path_id]
                if not having_at_least_one_path_flag:
                    try:
                        each_wk_k_u_path_ids[wk_idx][k][user_pair] = []
                    except:
                        try:
                            each_wk_k_u_path_ids[wk_idx][k]={}
                            each_wk_k_u_path_ids[wk_idx][k][user_pair] = []
                        except:
                            each_wk_k_u_path_ids[wk_idx]={}
                            each_wk_k_u_path_ids[wk_idx][k]={}
                            each_wk_k_u_path_ids[wk_idx][k][user_pair] = []
        
        
        """we set the required EPR pairs to achieve each fidelity threshold"""
        network.purification.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)       
        
        egr = solver.CPLEX_maximizing_EGR(wk_idx,network,0,0)
        return egr
    
    
    def set_paths_from_action(self,action,wk_idx,network,testing_flag):
        """this function uses the information in the chromosome 
        to set the paths to the data structure that will be used by solver"""
        def check_any_version_of_the_path_has_been_added(path_id,path_ids,path_versions):
            for path in path_versions:
                if path in path_ids:
                    return False
            return True
        """we set the required EPR pairs to achieve each fidelity threshold"""
        network.purification.set_required_EPR_pairs_for_distilation(wk_idx,network)
        network.each_wk_each_k_each_user_pair_id_paths={}
        network.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
#         network.each_wk_each_k_user_pair_ids = {}
#         network.each_wk_k_u_weight = {}
#         network.each_wk_k_u_pair_weight = {}
        path_indx = 0
        each_wk_k_u_path_ids = {}
        list_of_organizations = []
        for k in network.each_wk_organizations[wk_idx]:
            list_of_organizations.append(k)
        list_of_organizations.sort()
        for k in list_of_organizations:
            #print("training for k ",k)
            list_of_user_pairs = []
            #print("network.each_wk_each_k_user_pair_ids",network.each_wk_each_k_user_pair_ids)
            for user_pair_id in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
                list_of_user_pairs.append(user_pair_id)
            list_of_user_pairs.sort()
            for user_pair_id in list_of_user_pairs:
                #print("training user pair id %s from %s "%(user_pair_id,len(network.each_wk_each_k_user_pair_ids[wk_idx][k])))
                paths = network.each_user_pair_id_all_paths[user_pair_id]
                path_ids = []
                for path_id in paths:
                    #print("training for path id %s from %s"%(path_id,len(paths)))
                    if path_id in action:
                        path_ids.append(path_id)
                
                new_path_ids = []
                each_path_versions ={}
                for path_id in path_ids:
                    each_path_versions[path_id] = [path_id]
                    if path_id!=-1:
                        for sec_v_path_id in path_ids:
                            if path_id!= sec_v_path_id:
                                if sec_v_path_id in network.purification.each_path_version_numbers[path_id]:
                                    try:
                                        each_path_versions[path_id].append(sec_v_path_id)
                                    except:
                                        each_path_versions[path_id]=[sec_v_path_id]
                for path_id,version_path_ids in each_path_versions.items():
                    edge_g_value = network.purification.each_path_edge_level_g[path_id]
                    edge_level_g_values = []
                    for path_id_v in version_path_ids:
                        edge_level_g_values.append(network.purification.each_path_edge_level_g[path_id_v])
                    if edge_g_value >= max(edge_level_g_values):
                        if path_id not in new_path_ids:
                            if check_any_version_of_the_path_has_been_added(path_id,new_path_ids,network.purification.each_path_version_numbers[path_id]):
                                if network.purification.check_path_distilability(path_id):
                                    new_path_ids.append(path_id)

                path_ids = new_path_ids
                
                
                
                
                
                not_even_one_path = False
                path_counter_for_this_flow = 0
                for path_id in path_ids:
                    #print("training for path id %s from %s"%(path_id,len(paths)))
                    if path_id in action and path_counter_for_this_flow <network.num_of_paths:
                        try:
                            if path_id not in each_wk_k_u_path_ids[wk_idx][k][user_pair_id]:
                                path_counter_for_this_flow+=1
                                not_even_one_path = True
                                k = network.each_user_organization[user_pair_id]
                                try:
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id].append(path_id)
                                except:
                                    try:
                                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                    except:
                                        try:
                                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[path_id]
                                        except:
                                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[path_id]
                                            
                            else:
                                pass
                        except:
                            path_counter_for_this_flow+=1
                            not_even_one_path = True
                            k = network.each_user_organization[user_pair_id]
                            try:
                                network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id].append(path_id)
                            except:
                                try:
                                    network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                except:
                                    try:
                                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[path_id]
                                    except:
                                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[path_id]

                if not not_even_one_path:
                    try:
                        network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                    except:
                        try:
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[]
                        except:
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[]
                
     
       
#         return each_wk_k_u_path_ids

    
    def reward(self, wk_idx,network, actions,solver):
        """computes the reward. It first uses the action to set the paths in the network and then call the solvers"""
        chosen_paths = []
        for item in actions:
            chosen_paths.append(item)
        chosen_paths.sort()
        
#         print("we compute the reward ")
#         print("in reward we have these paths **************************** ",chosen_paths)
        try:
            if wk_idx in self.each_wk_action_reward:
                if tuple(chosen_paths) in self.each_wk_action_reward[wk_idx]:
                    rl_egr_value = self.each_wk_action_reward[wk_idx][tuple(chosen_paths)]
                else:
                    #print("1 goint to set paths from actions ")
                    self.set_paths_from_action(chosen_paths,wk_idx,network,False)
                    rl_egr_value  = solver.CPLEX_maximizing_EGR(wk_idx,network,2,2)
                    self.each_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr_value
            else:
                #print("2 goint to set paths from actions ")
                self.set_paths_from_action(chosen_paths,wk_idx,network,False)
                rl_egr_value  = solver.CPLEX_maximizing_EGR(wk_idx,network,2,2)
                self.each_wk_action_reward[wk_idx] = {}
                self.each_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr_value
        except:
            #print("3 goint to set paths from actions ")
            self.set_paths_from_action(chosen_paths,wk_idx,network,False)
            rl_egr_value  = solver.CPLEX_maximizing_EGR(wk_idx,network,2,2)
            self.each_wk_action_reward[wk_idx] = {}
            self.each_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr_value
        #print("**************  reward during training is ************** ",rl_egr_value)
        return rl_egr_value
    def find_combos(self,arr,k):
        combos = list(combinations(arr, k))
        return combos
    def get_all_possible_actions(self,each_user_paths,k):
        all_user_paths = []
        for user,paths in each_user_paths.items():
            paths = self.find_combos(paths,k)
            if paths:
                all_user_paths.append(paths)
        all_possible_actions = list(itertools.product(*all_user_paths))
        return all_possible_actions
    def compute_optimal_egr(self,wk_idx,network,solver):
        #print("computing optimal")
        max_egr = 0
        network.each_wk_each_k_each_user_pair_id_paths = {}
        each_user_pair_paths = {}
        for k,user_pair_ids in network.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair in user_pair_ids:
                each_user_pair_paths[user_pair] = network.each_pair_paths[user_pair]
                #print("we have %s for user pair %s "%(network.each_pair_paths[user_pair],user_pair))
        all_possible_actions = self.get_all_possible_actions(each_user_pair_paths,network.num_of_paths)
        #print("we have %s possible solutions in the optimal brute foce"%(len(all_possible_actions)))
        for action in all_possible_actions:
            actions = []
            for item in action:
                for i in item:
                    actions.append(i)
            egr = self.compute_egr(actions,wk_idx,network,solver)
            #print("action %s in the process of optimal search gave us %s "%(actions,egr))
            if egr >max_egr:
                max_egr = egr
                optimal_paths = actions
        if max_egr>0:
            return max_egr,optimal_paths
        else:
            return 0,[]
    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)
    def evaluate(self,wk_idx,network,solver,scheme,actions):
        if scheme =="RL":
            chosen_paths = []
            for item in actions:
                chosen_paths.append(item)
            chosen_paths.sort()
            self.set_paths_from_action(chosen_paths,wk_idx,network,True)
            
            rl_egr_value  = solver.CPLEX_maximizing_EGR(wk_idx,network,2,2)
#             self.each_testing_wk_action_reward[wk_idx] = {}
#             self.each_testing_wk_action_reward[wk_idx][tuple(chosen_paths)] = rl_egr_value

            return rl_egr_value
            
                
                


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:





# In[ ]:





# In[41]:





# In[ ]:




