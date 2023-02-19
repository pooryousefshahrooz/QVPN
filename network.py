#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
from itertools import islice
import matplotlib.pyplot as plt
import random
from itertools import groupby
import time
import math as mt
import csv
import os
import random
from solver import Solver
from genetic_algorithm import Genetic_algorithm
from reinforce import RL

import pdb
from os import listdir
from os.path import isfile, join
from threading import Thread


# In[ ]:





# In[ ]:





# In[ ]:


class Network:
    def __init__(self,config,purification_object ,topology_file,edge_capacity_bound,training_flag):
        self.data_dir = './data/'
        self.topology_file = self.data_dir+topology_file
        self.topology_name = topology_file
        self.toplogy_wk_scheme_result  = config.toplogy_wk_scheme_result_file
        self.training = training_flag
        self.set_E = []
        self.each_id_pair ={}
        self.pair_id = 0
        self.number_of_flows = 1

        self.min_edge_fidelity = float(config.min_edge_fidelity)
        self.max_edge_fidelity = float(config.max_edge_fidelity)
        self.num_of_paths = int(config.num_of_paths)
        
        
        # this is for checking the results of what epoch should be used in genetic algorthm initialization
        self.genetic_algorithm_initial_population_rl_epoch_number =config.genetic_algorithm_initial_population_rl_epoch_number
        
        
        # we use this to get the results of the training with how many training work loads
        self.number_of_training_wks = config.number_of_training_wks
        
        # this is the file that have the results fo rl training after different numbers of training epoch and we use for initialization the genetic algorithm        
        self.toplogy_wk_rl_for_initialization_ga_result_file = config.toplogy_wk_rl_for_initialization_ga_result_file
        self.number_of_flow_set = config.number_of_flow_set
        
        self.each_pair_id  ={}
        self.each_edge_distance = {}
        self.set_of_paths = {}
        self.each_u_paths = {}
        
        self.each_u_all_real_paths = {}
        self.each_u_all_real_disjoint_paths = {}
        self.each_u_paths = {}
        self.nodes = []
        
        self.path_counter_id = 0
        self.pair_id = 0
        self.q_value = 1
        self.each_node_q_value = {}
        self.each_u_weight={}
        self.each_path_legth = {}
        self.K= []
        self.each_k_u_all_paths = {}
        self.each_k_u_all_disjoint_paths={}
        self.each_wk_each_k_each_user_pair_id_paths = {}
        self.number_of_user_pairs = int(config.number_of_user_pairs)
        self.num_of_organizations = int(config.num_of_organizations)
        
        self.each_k_path_path_id = {}
        self.each_wk_each_k_user_pairs = {}
        self.each_wk_each_k_user_pair_ids = {}
        self.each_user_pair_all_paths = {}
        self.each_k_weight = {}
        self.each_k_u_weight = {}
        self.each_wk_k_u_pair_weight = {}
        self.each_pair_paths = {}
        self.each_scheme_each_user_pair_paths = {}
        self.each_user_organization = {}
        self.each_wk_organizations={}
        self.each_testing_user_organization = {}
        
        
        self.path_variables_file_path = config.flow_path_values_results
        
        # the value of multiplexing_value
        self.multiplexing_value = config.multiplexing_value
        
        # we test the rl scheme with these many work loads
        self.number_of_training_wks = config.number_of_training_wks
        
        # this is for checking which scheme is being used for path selection
        self.running_path_selection_scheme = "Hop"
        # we set how many workloads we will check the path selection scheme with
        self.workloads_to_test = config.workloads_to_test
        # data structures used for testing the reinforcement learnig scheme
        self.testing_work_loads=[]
        self.each_testing_wk_each_k_user_pairs={}
        self.each_testing_wk_each_k_user_pair_ids ={}
        self.each_testing_wk_k_weight = {}
        self.each_testing_wk_organizations = {}
        self.each_testing_wk_k_u_weight= {}
        self.each_testing_user_organization = {}
        self.each_testing_wk_k_u_pair_weight ={}
        
        self.max_edge_capacity = 0
        self.valid_flows =[]
        self.all_flows = []
        self.alpha_value = 1
        self.setting_basic_fidelity_flag = False
        self.each_link_cost_metric = "Hop"
        self.link_cost_metrics = []
        for scheme in config.schemes:
            if scheme in ["EGR","Hop","EGRSquare"]:
                self.link_cost_metrics.append(scheme)
                
        self.cut_off_for_path_searching = int(config.cut_off_for_path_searching)
        
        #we use the object of purification for taking of purificaiton and fidelity parameters
        self.purification = purification_object
        
        
        # we keep track of virtual links in order to set their weights to zero in path computation
        self.set_of_virtual_links = []
        # we load the topology 
        self.load_topology(edge_capacity_bound)
        
        
        
    def evaluate_rl_for_path_selection(self,config):
        """this function implements the main work flow of the reinforcement learning algorithm"""
        rl = RL(config)
        solver = Solver()
        self.each_link_cost_metric ="Hop" 
        self.set_link_weight("Hop")
        Thread(target = rl.train, args=(config,self,)).start()
        time.sleep(5)
        Thread(target = rl.test, args=(config,self,)).start()

                
    def evaluate_shortest_path_routing(self,config,link_cost_metric):
        """this function evaluates the entanglement generation rate using 
        shortest paths computed based on the given link cost metric"""
        self.each_wk_each_k_each_user_pair_id_paths = {}
        solver = Solver()
        self.each_link_cost_metric =link_cost_metric 
        self.set_link_weight(link_cost_metric)
        for wk_idx in self.work_loads:

            self.set_paths_in_the_network(wk_idx)
            """we set the required EPR pairs to achieve each fidelity threshold"""
            self.purification.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)       
            # calling the IBM CPLEX solver to solve the optimization problem
            
            egr = solver.CPLEX_maximizing_EGR(wk_idx,self,0,0)
                
            #network_capacity = solver.CPLEX_swap_scheduling(wk_idx,self)
            network_capacity= None
            print("for top %s work_load %s alpha_value %s metric %s number of paths %s we have egr as %s "%
                  (self.topology_name,wk_idx,self.alpha_value,link_cost_metric,self.num_of_paths,egr))
            self.save_results(wk_idx,config,False,False,True,None,0,egr,0,0)
            
            
    def get_each_user_all_paths(self,wk_idx,rl_testing_flag):
        """this function will set all the paths of each user pair for the given work load"""
        
#         if rl_testing_flag:
#             print("*********************** getting all user pairs all paths of workload %s ***************"%(wk_idx))
        self.each_user_pair_all_paths = {}
        if rl_testing_flag:
            """this part is for the user pairs that exist only in the testing workload but are not in trainign workload"""
            for k, user_pair_ids in self.each_testing_wk_each_k_user_pair_ids[wk_idx].items():
                #print("we have these user pairs %s in work load %s "%(user_pair_ids,wk_idx))
                for user_pair_id in user_pair_ids:
                    self.each_user_pair_all_paths[user_pair_id]= []
                    for path_id in self.each_pair_paths[user_pair_id]:
                        try:
                            self.each_user_pair_all_paths[user_pair_id].append(path_id)
                        except:
                            self.each_user_pair_all_paths[user_pair_id] = [path_id]
        else:
            for k, user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
                for user_pair_id in user_pair_ids:
                    self.each_user_pair_all_paths[user_pair_id]= []
                    for path_id in self.each_pair_paths[user_pair_id]:
                        try:
                            self.each_user_pair_all_paths[user_pair_id].append(path_id)
                        except:
                            self.each_user_pair_all_paths[user_pair_id] = [path_id]
                        

            
                        
                        
                        
    def set_paths_from_chromosome(self,wk_idx,chromosome):
        """this function uses the information in the chromosome 
        to set the paths to the data structure that will be used by solver"""
        path_indx = 0
        for k in self.each_wk_organizations[wk_idx]:
            for user_pair_id in self.each_wk_each_k_user_pair_ids[wk_idx][k]:
                if user_pair_id not in self.valid_flows:
                    try:
                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                    except:
                        try:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                        except:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=[]
                else:
                    path_ids = chromosome[path_indx:path_indx+self.num_of_paths]
                    path_ids = list(set(path_ids))
                    path_indx = path_indx+self.num_of_paths
                    k = self.each_user_organization[user_pair_id]
                    try:
                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = path_ids
                    except:
                        try:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = path_ids
                        except:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id]=path_ids
        
        #self.purificaiton.set_each_path_basic_fidelity()
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.purification.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)
        
    def save_rl_results_for_genetic_initialization(self,config,wk_idx,epoch_number,egr):
        """we save the paths that the rl suggests for this work load and use ot later 
        to initialize the genetic algorthm first population"""
        with open(self.toplogy_wk_rl_for_initialization_ga_result_file, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                for k in self.each_wk_organizations[wk_idx]:
                    for u in self.each_wk_each_k_user_pair_ids[wk_idx][k]: 
                        for p in self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]:
                            newFileWriter.writerow([self.topology_name,wk_idx,self.num_of_paths,
                            self.fidelity_threshold_range,self.q_value,
                            self.each_link_cost_metric,self.number_of_flows,config.number_of_training_wks,
                                egr,epoch_number,
                                config.cut_off_for_path_searching,k,u,p,self.set_of_paths[p]])
        
        
    def save_results(self,wk_idx,config,genetic_alg_flag,rl_flag,shortest_path_flag,genetic_alg,runs_of_algorithm,egr,optimal_egr,run_number):
        if self.end_level_purification_flag:
            purification_scheme = "End"
        else:
            purification_scheme = "Edge"
        if genetic_alg_flag:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                self.fidelity_threshold_range,purification_scheme,self.q_value,
                "Genetic",self.number_of_flows,genetic_alg.elit_pop_size,
                                        genetic_alg.crossover_p,genetic_alg.mutation_p,runs_of_algorithm,
                                        egr,genetic_alg.number_of_chromosomes,run_number,
                                        config.genetic_algorithm_random_initial_population,
                                        config.ga_elit_pop_update_step,config.ga_crossover_mutation_update_step,
                                        config.cut_off_for_path_searching,config.multi_point_mutation_value,
                                        config.multi_point_crossover_value,
                                        config.ga_crossover_mutation_multi_point_update_step,
                                        config.genetic_algorithm_initial_population_rl_epoch_number,
                                       config.number_of_training_wks,config.genetic_algorithm_initial_population])
        elif shortest_path_flag:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                self.fidelity_threshold_range,purification_scheme,self.q_value,
                self.each_link_cost_metric,self.number_of_flows,0,0,
                                        0,1,
                                        egr,0,run_number,
                                        0,
                                        0,0,
                                        config.cut_off_for_path_searching,0,
                                        0,0])
        elif rl_flag:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                self.fidelity_threshold_range,purification_scheme,self.q_value,
                "RL",self.number_of_flows,0,0,
                                        0,runs_of_algorithm,
                                        egr,0,run_number,
                                        0,
                                        0,0,
                                        config.cut_off_for_path_searching,0,
                                        0,0,config.rl_batch_size,config.initial_learning_rate,
                                       config.learning_rate_decay_rate,
                                       config.moving_average_decay,config.learning_rate_decay_step_multiplier,
                                        config.learning_rate_decay_step,
                                       config.entropy_weight,config.optimizer,config.scale,
                                        config.max_step,config.number_of_training_wks])
        
#                                     
                
    def evaluate_genetic_algorithm_for_path_selection(self,config):
        """this function implements the main work flow of the genetic algorithm"""
        solver = Solver()
        self.each_link_cost_metric ="Hop" 
        self.set_link_weight("Hop")
        runs_of_genetic_algorithm = 0
        genetic_alg = Genetic_algorithm(config)
        max_runs_of_genetic_algorithm = 1 # maximum number of populations during genetic algorithm search
        for wk_idx in self.work_loads:# Each work load includes a different set of user pairs in the network
            for elit_pop_size in genetic_alg.elit_pop_sizes:# percentage of top chromosomes that we generate next population
                genetic_alg.elit_pop_size = elit_pop_size

                for cross_over_value in genetic_alg.cross_over_values:# probability of applying crossover
                    genetic_alg.crossover_p = cross_over_value
                    for mutation_op_value in genetic_alg.mutation_op_values:# probability of applying mutation
                        genetic_alg.mutation_p =mutation_op_value 
                        for population_size in genetic_alg.population_sizes:
                            genetic_alg.number_of_chromosomes = population_size
                            """we set the set of all paths (all n shortest paths using different link cost metrics)"""
                            self.get_each_user_all_paths(wk_idx,False)
                            for i in range(config.runs_of_genetic_algorithm):
                                # we print the path ids for each user pair id
                                self.each_user_organization = {}
                                genetic_alg.generate_chromosomes(wk_idx,self)
                                max_fitness_value = 0
                                best_chromosome = ""
                                genetic_algorithm_running_flag = True
                                runs_of_genetic_algorithm = 0
                                while(runs_of_genetic_algorithm < genetic_alg.max_runs_of_genetic_algorithm):
                                    genetic_alg.each_fitness_chromosomes = {}
                                    chromosome_id = 0
                                    for chromosome in genetic_alg.chromosomes:
                                        self.set_paths_from_chromosome(wk_idx,chromosome)
                                        fitness_value  = solver.CPLEX_maximizing_EGR(wk_idx,self,runs_of_genetic_algorithm,chromosome_id)
                                        fitness_value = round(fitness_value,3)
                                        try:
                                            genetic_alg.each_fitness_chromosomes[fitness_value].append(chromosome)
                                        except:
                                            genetic_alg.each_fitness_chromosomes[fitness_value] = [chromosome]

                                        # we save the max egr at this point of genetic algorithm
                                        self.save_results(wk_idx,config,True,False,False,genetic_alg,runs_of_genetic_algorithm,fitness_value,0,i)
                                        genetic_alg.update_operation_probabilities(runs_of_genetic_algorithm,config)
                                        chromosome_id+=1
                                        
                                        if runs_of_genetic_algorithm%50==0:
                                            
                                            print("for wk %s run %s topology %s flow size %s num paths %s chromosome %s th from %s we got egr %s step %s from %s"
                                              %(wk_idx,i,self.topology_name,self.number_of_flows,self.num_of_paths,chromosome_id,len(genetic_alg.chromosomes),fitness_value,
                                                runs_of_genetic_algorithm,genetic_alg.max_runs_of_genetic_algorithm))
                                            #print("******************************* one round of genetic algorithm remained %s*******************",genetic_alg.max_runs_of_genetic_algorithm-runs_of_genetic_algorithm)
                                    
                                    genetic_alg.population_gen_op()
                                    print("runs_of_genetic_algorithm %s from %s "%(runs_of_genetic_algorithm,genetic_alg.max_runs_of_genetic_algorithm))
                                    runs_of_genetic_algorithm+=1
    
    
   
                                    
    def load_topology(self,each_edge_capacity_upper_bound):
        self.set_E=[]
        self.each_edge_capacity={}
        self.nodes = []
        self.purification.each_edge_fidelity = {}
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        edge_counter = 0
        self.g = nx.Graph()
        print('[*] Loading topology...', self.topology_file)
        try:
            f = open(self.topology_file+".txt", 'r')
        except:
            f = open(self.topology_file, 'r')
        header = f.readline()
        for line in f:
            edge_counter+=1
            line = line.strip()
            link = line.split('\t')
            #print(line,link)
            i, s, d,  c,l,real_virtual = link
            if real_virtual =="virtual":
                self.set_of_virtual_links.append((int(s),int(d)))
                self.set_of_virtual_links.append((int(d),int(s)))
            if int(s) not in self.nodes:
                self.nodes.append(int(s))
            if int(d) not in self.nodes:
                self.nodes.append(int(d))
            self.set_E.append((int(s),int(d)))
            self.max_edge_capacity = each_edge_capacity_upper_bound
            edge_capacity = float(c)
            self.each_edge_distance[(int(s),int(d))] = float(l)
            self.each_edge_distance[(int(d),int(s))] = float(l)
            self.each_edge_capacity[(int(s),int(d))] = edge_capacity
            self.each_edge_capacity[(int(d),int(s))] = edge_capacity
            if (int(s),int(d)) in self.g.edges:
                print("duplicate  edge ",int(s),int(d))
            if real_virtual =="virtual":
                self.g.add_edge(int(s),int(d),capacity=edge_capacity,weight=1)
                self.g.add_edge(int(d),int(s),capacity=edge_capacity,weight=1)
            else:
                self.g.add_edge(int(s),int(d),capacity=edge_capacity,weight=1)
                self.g.add_edge(int(d),int(s),capacity=edge_capacity,weight=1)
        f.close()
        print("*********            # edges %s edge_counter %s self.set_E %s ************** "%(len(self.g.edges),edge_counter,len(self.set_E)))

        
        
    def set_nodes_q_value(self):
        for node in self.nodes:
            self.each_node_q_value[node] = self.q_value
    def find_longest_path(self,source,destination):
        # Get the longest path from node source to node destination
        longest_path = max(nx.all_simple_paths(self.g, source, destination), key=lambda x: len(x))
        return longest_path
    
    def get_path_info(self):
        self.all_user_pairs_across_wks = []
        self.each_pair_paths = {}
        self.each_scheme_each_user_pair_paths = {}
        set_of_all_paths = []
        self.path_counter_id = 0
        self.path_counter = 0
        print("self.each_wk_organizations",self.each_wk_organizations)
        for wk,ks in self.each_testing_wk_organizations.items():
            for k in ks:
                try:
                    for u in self.each_testing_wk_each_k_user_pairs[wk][k]:
                        if u not in self.all_user_pairs_across_wks:
                            self.all_user_pairs_across_wks.append(u)
                except:
                    pass
                
        for wk,ks in self.each_wk_organizations.items():
            for k in ks:
                for u in self.each_wk_each_k_user_pairs[wk][k]:
                    if u not in self.all_user_pairs_across_wks:
                        #if self.running_path_selection_scheme=="RL":
                        self.all_user_pairs_across_wks.append(u)

        for link_cost_metric in self.link_cost_metrics:
            for user_pair in self.all_user_pairs_across_wks:
                user_pair_id = self.each_pair_id[user_pair]
                self.each_pair_paths[user_pair_id]=[]
        for user_pair in self.all_user_pairs_across_wks:
            having_atleast_one_path_flag = False
            for link_cost_metric in ["Hop","EGR","EGRSquare"]:
#             for link_cost_metric in ["Hop"]:#just for now to fix the isuue with iniializing the genetic algorthm
                self.each_link_cost_metric =link_cost_metric 
                self.set_link_weight(link_cost_metric)
                user_pair_id = self.each_pair_id[user_pair]
                paths = self.get_paths_between_user_pairs(user_pair)
                selected_path_for_this_scheme = []
                path_flag = False
                for path in paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):                
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    
                    if path_edges not in set_of_all_paths:
                        path_fidelity  = self.purification.get_fidelity(path_edges,self.set_of_virtual_links)
                        if path_fidelity>0.50:
                            self.purification.each_path_basic_fidelity[self.path_counter_id]= round(path_fidelity,3)
                            path_flag= True
                            having_atleast_one_path_flag = True
                            set_of_all_paths.append(path_edges)
                            self.set_each_path_length(self.path_counter_id,path_edges)
                            self.set_of_paths[self.path_counter_id] = path_edges
                            if self.path_counter_id not in selected_path_for_this_scheme:
                                selected_path_for_this_scheme.append(self.path_counter_id)
                            try:
                                self.each_pair_paths[user_pair_id].append(self.path_counter_id)
                            except:
                                self.each_pair_paths[user_pair_id] = [self.path_counter_id]
                            self.path_counter_id+=1  
                            self.path_counter+=1
                    
                        
                #if link_cost_metric=="Hop":
                    #print("scheme %s flow %s have these path ids %s "%(link_cost_metric,user_pair_id,selected_path_for_this_scheme))
                    #time.sleep(1)
                try:
                    self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=selected_path_for_this_scheme
                except:
                    self.each_scheme_each_user_pair_paths[link_cost_metric]={}
                    self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id] = selected_path_for_this_scheme
                    
                if not path_flag:# the flow does not have a valid path in this scsheme (path with fidelity higher than 0.6)
                    try:
                        self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=[]
                    except:
                        self.each_scheme_each_user_pair_paths[link_cost_metric]={}
                        self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id] = []
            if not having_atleast_one_path_flag:#the flow does not have a valid path in this scsheme (path with fidelity higher than 0.6 using any scheme)
                self.each_pair_paths[user_pair_id] = []
    def get_paths_between_user_pairs(self,user_pair):
        
        return self.k_shortest_paths(user_pair[0], user_pair[1], self.cut_off_for_path_searching,"weight")
    
    def check_edge_exit(self,edge):
        if edge in self.set_E:
            return 1
        else:
            return 0
    
            
    def set_link_weight(self,link_cost_metric):
        for edge in self.g.edges:
            edge_capacity = self.each_edge_capacity[edge]
            if edge in self.set_of_virtual_links or (edge[1],edge[0]) in self.set_of_virtual_links:
                weight1=0
                weight2 = 0
            elif link_cost_metric =="Hop":
                weight1=1
                weight2 = 1
            elif link_cost_metric =="EGR":
                weight1=1/edge_capacity
                weight2 = 1/edge_capacity
            elif link_cost_metric =="EGRSquare":
                weight1=1/(edge_capacity**2)
                weight2 = 1/(edge_capacity**2)
            elif link_cost_metric =="Bruteforce":
                weight1=1
                weight2 = 1
            self.g[edge[0]][edge[1]]['weight']=weight1
            self.g[edge[1]][edge[0]]['weight']= weight2
            
    def update_link_rates(self,alpha_value):
        self.alpha_value = alpha_value
        edge_rates = []
        real_edges_rates = []# this is becasue we want to set the capacity of virtual links to the highest possible rate
        for edge in self.g.edges:
            edge_length = self.each_edge_distance[edge]
            if edge not in self.set_of_virtual_links and (edge[1],edge[0]) not in self.set_of_virtual_links:
                c = 1
                etha = 10**(-0.1*0.2*edge_length)
                T = (edge_length*10**(-4))/25# based on simulation setup of data link layer paper
                edge_rate = self.multiplexing_value*(2*c*etha*alpha_value)/T
                real_edges_rates.append(edge_rate)
        for edge in self.g.edges:
            edge_length = self.each_edge_distance[edge]
            if edge in self.set_of_virtual_links or (edge[1],edge[0]) in self.set_of_virtual_links:# if it is a virtual link, then set it to highest capacity
                edge_rate  = max(real_edges_rates)+1
                edge_fidelity = 1 # means virtual links have fidleity 1!
            else:
                c = 1
                etha = 10**(-0.1*0.2*edge_length)
                T = (edge_length*10**(-4))/25# based on simulation setup of data link layer paper
                edge_rate = self.multiplexing_value*(2*c*etha*alpha_value)/T
                edge_fidelity = 1-alpha_value
            self.g[edge[0]][edge[1]]['capacity']=edge_rate
            self.each_edge_capacity[(edge[0],edge[1])] = edge_rate
            self.each_edge_capacity[(edge[1],edge[0])] = edge_rate
            self.purification.each_edge_fidelity[edge] = edge_fidelity
            self.purification.each_edge_fidelity[(edge[1],edge[0])] = edge_fidelity
            edge_rates.append(edge_rate)
            #print("alpha %s generated %s with fidelity %s "%(alpha_value,edge_rate,1-alpha_value))
            
        self.max_edge_capacity = max(edge_rates)
        
    def set_flows_of_organizations(self):
        """This function reads the workload from the topology+WK file and set the data structures"""
        self.each_wk_k_weight = {}
        self.each_wk_organizations={}
        self.each_wk_each_k_user_pair_ids = {}
        self.each_wk_each_k_user_pairs = {}
        self.pair_id = 0
        self.each_id_pair ={}
        self.each_pair_id={}
        self.work_loads=[]
        self.each_wk_k_u_weight ={}
        self.each_wk_k_u_pair_weight = {}
        
        """data structures used for testing the reinforcement learning scheme"""
        self.testing_work_loads=[]
        self.each_testing_wk_each_k_user_pairs={}
        self.each_testing_wk_each_k_user_pair_ids ={}
        self.each_testing_wk_k_weight = {}
        self.each_testing_wk_k_u_weight= {}
        self.each_testing_wk_organizations = {}
        self.each_testing_wk_k_u_pair_weight ={}
        
        num_nodes = len(self.nodes)
        try:
            work_load_file = self.topology_file.split(".txt")[0]
        except:
            if ".txt" not in self.topology_file:
                work_load_file = self.topology_file
        if self.running_path_selection_scheme in ["EGR","EGRSquare","Hop","Genetic"]:
            f = open(work_load_file+"WK2", 'r')
        else:
            f = open(work_load_file+"WK", 'r')
        self.work_load_counter = 0
        all_active_user_pairs_acros_wks = []
        header = f.readline()
        for line in f:
            
            values = line.strip().split(',')#wk_indx,organization,weight,user_pair,weight
            wk_idx = int(values[0])
            k = int(values[1])
            if k < self.num_of_organizations and (wk_idx<self.workloads_to_test or (self.running_path_selection_scheme =="RL" and wk_idx < self.number_of_training_wks)):
                org_weight = float(values[2])
                organization_F = float(values[3])
                i = int(values[4].split(":")[0])
                j = int(values[4].split(":")[1])
                flow_weight = float(values[5])
                flow_fidelity_threshold = float(values[6])
                user_pair = (i,j)
                flow_number_set = int(values[7])
                if flow_number_set ==self.number_of_flows:
                    if wk_idx not in self.work_loads:
                        self.work_loads.append(wk_idx)

                    """we check how many flows we have set for each organization"""   
                    try:
                        num_covered_flows = len(self.each_wk_each_k_user_pairs[wk_idx][k])
                    except:
                        num_covered_flows = 0
                    if num_covered_flows<self.number_of_flows:
                        try:
                            user_pair_id = self.each_pair_id[user_pair] 
                        except:
                            user_pair_id = self.pair_id
                            self.pair_id+=1

                        try:
                            self.each_wk_each_k_user_pairs[wk_idx][k].append(user_pair)
                        except:
                            self.each_wk_each_k_user_pairs[wk_idx]={}
                            self.each_wk_each_k_user_pairs[wk_idx][k]=[user_pair]
                        """we create an id for this flow and added to the data structure
                        This is becasue we want to let two organizations have same flows but with different ids"""
                        self.each_id_pair[user_pair_id] = user_pair
                        self.each_pair_id[user_pair] = user_pair_id
                        self.each_user_organization[user_pair_id] = k
                        
                        # we set the fidelity threshold of the flow
                        try:
                            self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id] =flow_fidelity_threshold 
                        except:
                            try:
                                self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k]={}
                                self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id] =flow_fidelity_threshold 
                            except:
                                self.purification.each_wk_k_u_fidelity_threshold[wk_idx]={}
                                self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k]={} 
                                self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id] =flow_fidelity_threshold 
                                
                        
#                         print(" **** adding pair %s # %s for work load %s for user pair %s ****"%(user_pair_id,num_covered_flows,wk_idx,user_pair))
                        try:
                            self.each_wk_each_k_user_pair_ids[wk_idx][k].append(user_pair_id)
                        except:
                            try:
                                self.each_wk_each_k_user_pair_ids[wk_idx][k]= [user_pair_id]
                            except:
                                self.each_wk_each_k_user_pair_ids[wk_idx]={}
                                self.each_wk_each_k_user_pair_ids[wk_idx][k]= [user_pair_id]
                        # we set the weight of each flow
                        try:
                            self.each_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight
                            self.each_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight
                        except:
                            try:
                                self.each_wk_k_u_weight[wk_idx][k] ={}
                                self.each_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight

                                self.each_wk_k_u_pair_weight[wk_idx][k] ={}
                                self.each_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight
                            except:
                                self.each_wk_k_u_weight[wk_idx] ={}
                                self.each_wk_k_u_weight[wk_idx][k] = {}
                                self.each_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight

                                self.each_wk_k_u_pair_weight[wk_idx] ={}
                                self.each_wk_k_u_pair_weight[wk_idx][k] ={}
                                self.each_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight

                        
                        """We set the weight of the organization"""
                        try:
                            self.each_wk_k_weight[wk_idx][k] = org_weight
                        except:
                            self.each_wk_k_weight[wk_idx]= {}
                            self.each_wk_k_weight[wk_idx][k] = org_weight  
                        # we set the work load its organization
                        try:
                            if k not in self.each_wk_organizations[wk_idx]:
                                try:
                                    self.each_wk_organizations[wk_idx].append(k)
                                except:
                                    self.each_wk_organizations[wk_idx]=[k]
                        except:
                            self.each_wk_organizations[wk_idx]=[k]
#                         print("***** we have work load %s for %s ***** "%(wk_idx,self.running_path_selection_scheme))
        
                    
        
        """this part is for reinforcement learning scheme.
        we involve the data for testing the reinforcment learning scheem in the organization and flow info
        
        in order to have same state action dimensions in training and testing """  
#         print("********************************** adding testing for RL",self.workloads_to_test,self.running_path_selection_scheme)
#         time.sleep(10)
        
        if self.running_path_selection_scheme in ["RL"]:
            f = open(work_load_file+"WK2", 'r')
            num_covered_flows = 0
            try:
                work_load_file = self.topology_file.split(".txt")[0]
            except:
                if ".txt" not in self.topology_file:
                    work_load_file = self.topology_file

            header = f.readline()
#             print("we are testing RL and we are adding testing workloads ",self.workloads_to_test)
            for line in f:
            
                values = line.strip().split(',')#wk_indx,organization,weight,user_pair,weight
                wk_idx = int(values[0])
                k = int(values[1])
                if wk_idx<self.workloads_to_test and k < self.num_of_organizations:
                    org_weight = float(values[2])
                    organization_F = float(values[3])
                    i = int(values[4].split(":")[0])
                    j = int(values[4].split(":")[1])
                    flow_weight = float(values[5])
                    flow_fidelity_threshold = float(values[6])
                    user_pair = (i,j)
                    flow_number_set = int(values[7])
                    try:
                        user_pair_id = self.each_pair_id[user_pair] 
                    except:
                        user_pair_id = self.pair_id
                        self.pair_id+=1
                    
                    if flow_number_set ==self.number_of_flows:
                        if wk_idx not in self.testing_work_loads:
                            self.testing_work_loads.append(wk_idx)

                        """we check how many flows we have set for each organization"""   
                        try:
                            num_covered_flows = len(self.each_testing_wk_each_k_user_pairs[wk_idx][k])
                        except:
                            num_covered_flows = 0
                        if num_covered_flows<self.number_of_flows:
                            try:
                                self.each_testing_wk_each_k_user_pairs[wk_idx][k].append(user_pair)
                            except:
                                self.each_testing_wk_each_k_user_pairs[wk_idx]={}
                                self.each_testing_wk_each_k_user_pairs[wk_idx][k]=[user_pair]

                            """we create an id for this flow and added to the data structure
                            This is becasue we want to let two organizations have same flows but with different ids"""
                            self.each_id_pair[user_pair_id] = user_pair
                            self.each_pair_id[user_pair] = user_pair_id
                            self.each_testing_user_organization[user_pair_id] = k
    #                         print(" **** adding pair %s # %s for work load %s for user pair %s ****"%(user_pair_id,num_covered_flows,wk_idx,user_pair))
                            try:
                                self.each_testing_wk_each_k_user_pair_ids[wk_idx][k].append(user_pair_id)
                            except:
                                try:
                                    self.each_testing_wk_each_k_user_pair_ids[wk_idx][k]= [user_pair_id]
                                except:
                                    self.each_testing_wk_each_k_user_pair_ids[wk_idx]={}
                                    self.each_testing_wk_each_k_user_pair_ids[wk_idx][k]= [user_pair_id]
                            # we set the weight of each flow
                            try:
                                self.each_testing_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight
                                self.each_testing_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight
                            except:
                                try:
                                    self.each_testing_wk_k_u_weight[wk_idx][k] ={}
                                    self.each_testing_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight

                                    self.each_testing_wk_k_u_pair_weight[wk_idx][k] ={}
                                    self.each_testing_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight
                                except:
                                    self.each_testing_wk_k_u_weight[wk_idx] ={}
                                    self.each_testing_wk_k_u_weight[wk_idx][k] = {}
                                    self.each_testing_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight

                                    self.each_testing_wk_k_u_pair_weight[wk_idx] ={}
                                    self.each_testing_wk_k_u_pair_weight[wk_idx][k] ={}
                                    self.each_testing_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight


                            """We set the weight of the organization"""
                            try:
                                self.each_testing_wk_k_weight[wk_idx][k] = org_weight
                            except:
                                self.each_testing_wk_k_weight[wk_idx]= {}
                                self.each_testing_wk_k_weight[wk_idx][k] = org_weight  
                            # we set the work load its organization
                            try:
                                if k not in self.each_testing_wk_organizations[wk_idx]:
                                    try:
                                        self.each_testing_wk_organizations[wk_idx].append(k)
                                    except:
                                        self.each_testing_wk_organizations[wk_idx]=[k]
                            except:
                                self.each_testing_wk_organizations[wk_idx]=[k]
                        
                    
                    
                    
    def set_each_k_user_pair_paths(self,wk_idx):
        """we set self.num_of_paths for each user pair of each organization """
        self.path_counter_id  = 0
        added_paths = []
        self.each_wk_each_k_each_user_pair_id_paths = {}
        for k,user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair_id in user_pair_ids:
                user_pair = self.each_id_pair[user_pair_id]
                having_at_least_one_path_flag = False
                one_path_added_flag = False
                for path in self.k_shortest_paths(user_pair[0], user_pair[1], self.num_of_paths,"weight"):
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    flag= False
                    for p_id,edges in self.set_of_paths.items():
                        if edges == path_edges:
                            path_id2 = p_id
                            flag = True
                    if flag:
                        path_id=path_id2
                    path_fidelity = self.purification.get_fidelity(path_edges,self.set_of_virtual_links)
                    if path_fidelity>0.5:# the condition of 1==1 is for the reasont hat we want to use all paths
                        if not flag:
                            path_id = self.path_counter_id
                            added_paths.append(path_edges)
                            self.set_each_path_length(self.path_counter_id,path_edges)
                            self.set_of_paths[self.path_counter_id] = path_edges
                                
                            #we set the basic fidelity of path here
                            self.purification.each_path_basic_fidelity[path_id]= round(path_fidelity,3)
                            try:
                                if len(self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id])<self.num_of_paths:
                                    having_at_least_one_path_flag = True
                                    one_path_added_flag=True
                                    try:
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id].append(path_id)
                                    except:
                                        try:
                                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                        except:
                                            try:
                                                self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                                self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                            except:
                                                self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                                self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                                self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                            except:
                                having_at_least_one_path_flag = True
                                one_path_added_flag=True
                                try:
                                    self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                except:
                                    try:
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                                    except:
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = [path_id]
                            self.path_counter_id+=1
                if not having_at_least_one_path_flag:
                    try:
                        self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                    except:
                        try:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []
                        except:
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][user_pair_id] = []

#                     shortest_disjoint_paths = nx.edge_disjoint_paths(self.g,s=user_pair[0],t=user_pair[1])             
        
    def k_shortest_paths(self, source, target, k, weight):
        return list(
            islice(nx.shortest_simple_paths(self.g, source, target, weight=weight), k)
        )
    
    
    
    
    def set_paths_in_the_network(self,wk_idx):
        self.reset_pair_paths()
        self.set_each_k_user_pair_paths(wk_idx)
        #self.purification.set_each_path_basic_fidelity()
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.purification.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)
        
    def reset_pair_paths(self):
        self.path_id = 0
        self.path_counter_id = 0
        self.set_of_paths = {}
        self.each_k_u_paths = {}
        self.each_k_u_disjoint_paths = {}
        self.each_k_path_path_id = {}
        self.each_k_u_all_paths = {}
        self.each_k_u_all_disjoint_paths = {}
        self.each_k_path_path_id={}
        
    def set_each_user_pair_demands(self):
        self.each_t_each_request_demand = {}
        num_of_pairs= len(list(self.user_pairs))
        tm = spike_tm(num_of_pairs+1,num_spikes,spike_mean,1)
        
        traffic = tm.at_time(1)
        printed_pairs = []
        user_indx = 0
        for i in range(num_of_pairs):
            for j in range(num_of_pairs):
                if i!=j:
                    if (i,j) not in printed_pairs and (j,i) not in printed_pairs and user_indx<num_of_pairs:
                        printed_pairs.append((i,j))
                        printed_pairs.append((j,i))
#                             print("num_of_pairs %s time %s traffic from %s to %s is %s and user_indx %s"%(num_of_pairs, time,i,j,traffic[i][j],user_indx))
                        request = user_pairs[time][user_indx]
                        user_indx+=1
                        demand = max(1,traffic[i][j])
                        try:
                            self.each_u_request_demand[request] = demand
                        except:
                            self.each_u_demand[request] = demand
        for request in self.user_pairs[time]:
            try:
                self.each_u_demand[0][request] = 0
            except:
                self.each_u_demand[0]={}
                self.each_u_demand[0][request] = 0
    
    

        
    def reset_variables(self):
        self.each_id_pair ={}
        self.pair_id = 0
        self.max_edge_capacity = int(config.max_edge_capacity)
        self.min_edge_capacity = int(config.min_edge_capacity)
        self.min_edge_fidelity = float(config.min_edge_fidelity)
        self.max_edge_fidelity = float(config.max_edge_fidelity)
        self.num_of_paths = int(config.num_of_paths)
        self.path_selection_scheme = config.path_selection_scheme
        self.each_pair_id  ={}
       
        self.set_of_paths = {}
        self.each_u_paths = {}
        self.purificaiton.each_n_f_purification_result = {}
        self.purificaiton.each_edge_target_fidelity = {}
        self.each_u_all_real_paths = {}
        self.each_u_all_real_disjoint_paths = {}
        self.each_u_paths = {}
        self.nodes = []
        self.purificaiton.oracle_for_target_fidelity = {}
        self.each_k_path_path_id = {}
        self.purificaiton.global_each_basic_fidelity_target_fidelity_required_EPRs = {}
        self.purificaiton.purificaiton.all_basic_fidelity_target_thresholds = []
        self.path_counter_id = 0
        self.pair_id = 0
        self.each_u_weight={}
        self.each_path_legth = {}
        self.load_topology()
    
    
        
    
    def set_each_path_length(self,path_id,path_edges):
        path_length = 0
        for edge in path_edges:
            if edge not in self.set_of_virtual_links and (edge[1],edge[0]) not in self.set_of_virtual_links:
                path_length+=1
        self.each_path_legth[path_id] = path_length
    
    
        
    def get_real_longest_path(self,user_or_storage_pair,number_of_paths):
        all_paths=[]
        for path in nx.all_simple_paths(self.g,source=user_or_storage_pair[0],target=user_or_storage_pair[1]):
            #all_paths.append(path)

            node_indx = 0
            path_edges = []
            for node_indx in range(len(path)-1):
                path_edges.append((path[node_indx],path[node_indx+1]))
                node_indx+=1
            all_paths.append(path_edges)

        all_paths.sort(key=len,reverse=True)
        if len(all_paths)>=number_of_paths:
            return all_paths[:number_of_paths]
        else:
            return all_paths
                        
    def get_real_path(self,user_or_storage_pair_id):
        if self.path_selection_scheme=="shortest":
            path_selecion_flag = False
            path_counter = 1
            paths = []
            #print("user_or_storage_pair",user_or_storage_pair)
            #print("self.each_user_pair_all_real_paths[user_or_storage_pair]",self.each_user_pair_all_real_paths[user_or_storage_pair])
            for path in self.each_user_pair_all_real_paths[user_or_storage_pair_id]:
                #print("we can add this path",path)
                if path_counter<=self.num_of_paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    paths.append(path_edges)

                path_counter+=1
        elif self.path_selection_scheme=="shortest_disjoint":
            path_selecion_flag = False
            path_counter = 1
            paths = []
            #print("self.each_user_pair_all_real_paths[user_or_storage_pair]",self.each_user_pair_all_real_paths[user_or_storage_pair])
            for path in self.each_user_pair_all_real_disjoint_paths[user_or_storage_pair_id]:
                #print("we can add this path",path)
                if path_counter<=self.num_of_paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    paths.append(path_edges)

                path_counter+=1
            
        return paths
                    
      
                    
    
   
    def get_edges(self):
        return self.set_E
    

    def check_path_include_edge(self,edge,path):
        
        if edge in self.set_of_paths[path]:
            return True
        elif edge not  in self.set_of_paths[path]:
            return False

    def check_request_use_path(self,k,p):
        if p in self.each_u_paths[k]:
            return True
        else:
            return False
    def get_path_length(self,path):
        return self.each_path_legth[path]-1

    


