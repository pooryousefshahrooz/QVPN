#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from train import RL
import pdb

from os import listdir

from os.path import isfile, join


# In[11]:


class Network:
    def __init__(self,config,topology_file,edge_capacity_bound,training_flag):
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
        
        self.number_of_flow_set = config.number_of_flow_set
        
        self.each_pair_id  ={}
        self.each_edge_distance = {}
        self.set_of_paths = {}
        self.each_u_paths = {}
        self.each_n_f_purification_result = {}
        self.each_edge_target_fidelity = {}
        self.each_u_all_real_paths = {}
        self.each_u_all_real_disjoint_paths = {}
        self.each_u_paths = {}
        self.nodes = []
        self.oracle_for_target_fidelity = {}
        self.global_each_basic_fidelity_target_fidelity_required_EPRs = {}
        self.all_basic_fidelity_target_thresholds = []
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
        self.each_wk_k_fidelity_threshold = {}
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
                
        self.cut_off_for_path_searching = max(int(config.cut_off_for_path_searching),config.num_of_paths)
        self.load_topology(edge_capacity_bound)
        
    def evaluate_rl_for_path_selection(self,config):
        rl = RL()
        rl.main(config,self)
    def evaluate_shortest_path_routing(self,link_cost_metric):
        """this function evaluates the entanglement generation rate using 
        shortest paths computed based on the given link cost metric"""
        self.each_wk_each_k_each_user_pair_id_paths = {}
        solver = Solver()
        self.each_link_cost_metric =link_cost_metric 
        self.set_link_weight(link_cost_metric)
        for wk_idx in self.work_loads:
            self.set_paths_in_the_network(wk_idx)
            
            if not self.setting_basic_fidelity_flag:
                self.set_each_path_basic_fidelity()
                self.setting_basic_fidelity_flag = True
                """we set the required EPR pairs to achieve each fidelity threshold"""
                self.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)       
            # calling the IBM CPLEX solver to solve the optimization problem
            egr = solver.CPLEX_maximizing_EGR(wk_idx,self)
        
            #network_capacity = solver.CPLEX_swap_scheduling(wk_idx,self)
            network_capacity= None
            print("for top %s work_load %s alpha_value %s metric %s number of paths %s we have egr as %s cepacity %s"%
                  (self.topology_name,wk_idx,self.alpha_value,link_cost_metric,self.num_of_paths,egr,network_capacity))
            self.save_results(wk_idx,False,None,0,egr)
            
            
    def get_each_user_all_paths(self,wk_idx):
        """this function will set all the paths of each user pair for the given work load"""
        self.each_user_pair_all_paths = {}
        for k, user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            #print("we have these user pairs %s in work load %s "%(user_pair_ids,wk_idx))
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
        
        self.set_each_path_basic_fidelity()
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)
        
    def save_results(self,wk_idx,genetic_alg_flag,genetic_alg,runs_of_genetic_algorithm,egr):
        if self.end_level_purification_flag:
            purification_scheme = "End"
        else:
            purification_scheme = "Edge"
        if genetic_alg_flag:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                self.fidelity_threshold_range,purification_scheme,self.q_value,
                "Genetic",self.number_of_flows,genetic_alg.elit_pop_size,genetic_alg.selection_p,
                                        genetic_alg.crossover_p,genetic_alg.mutation_p,runs_of_genetic_algorithm,
                                        egr])
        else:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                self.fidelity_threshold_range,purification_scheme,self.q_value,
                self.each_link_cost_metric,self.number_of_flows,0,0,
                                        0,0,1,
                                        egr])
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
                for selection_op_value in genetic_alg.selection_op_values: # probability of chosing from elit or all
                    genetic_alg.selection_p = selection_op_value
                    for cross_over_value in genetic_alg.cross_over_values:# probability of applying crossover
                        genetic_alg.crossover_p = cross_over_value
                        for mutation_op_value in genetic_alg.mutation_op_values:# probability of applying mutation
                            genetic_alg.mutation_p =mutation_op_value 
                            for population_size in genetic_alg.population_sizes:
                                genetic_alg.number_of_chromosomes = population_size
                                """we set the set of all paths (all n shortest paths using different link cost metrics)"""
                                self.get_each_user_all_paths(wk_idx)
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
                                        fitness_value  = solver.CPLEX_maximizing_EGR(wk_idx,self)
                                        fitness_value = round(fitness_value,3)
                                        try:
                                            genetic_alg.each_fitness_chromosomes[fitness_value].append(chromosome)
                                        except:
                                            genetic_alg.each_fitness_chromosomes[fitness_value] = [chromosome]
                                        # we store the best fitness value and the chromosome associated to it
                                       # in our final loop of genetic algorithm
                                        if runs_of_genetic_algorithm >= genetic_alg.max_runs_of_genetic_algorithm:
                                            genetic_algorithm_running_flag = False
                                            if fitness_value>max_fitness_value:
                                                max_fitness_value = fitness_value
                                                best_chromosome = chromosome
                                        chromosome_id+=1
                                        if runs_of_genetic_algorithm%100==0:
                                            print("for wk %s flow size %s chromosome %s th from %s we got egr %s step %s from %s"
                                                  %(wk_idx,self.number_of_flows,chromosome_id,len(genetic_alg.chromosomes),fitness_value,
                                                    runs_of_genetic_algorithm,genetic_alg.max_runs_of_genetic_algorithm))

                                    genetic_alg.population_gen_op()
                                    runs_of_genetic_algorithm+=1
                                    # we save the max egr at this point of genetic algorithm
                                    self.save_results(wk_idx,True,genetic_alg,runs_of_genetic_algorithm,fitness_value)
                                print("for work load %s we have entanglement generation rate of %s using these paths %s"%
                                      (wk_idx,max_fitness_value,best_chromosome))
    
    
   
                                    
    def load_topology(self,each_edge_capacity_upper_bound):
        self.set_E=[]
        self.each_edge_capacity={}
        self.nodes = []
        self.each_edge_fidelity = {}
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.g = nx.Graph()
        print('[*] Loading topology...', self.topology_file)
        try:
            f = open(self.topology_file+".txt", 'r')
        except:
            f = open(self.topology_file, 'r')
        header = f.readline()
        for line in f:
            line = line.strip()
            link = line.split('\t')
            #print(line,link)
            i, s, d,  c,l = link
            if int(s) not in self.nodes:
                self.nodes.append(int(s))
            if int(d) not in self.nodes:
                self.nodes.append(int(d))
            self.set_E.append((int(s),int(d)))
#             random_fidelity = random.uniform(self.min_edge_fidelity,self.max_edge_fidelity)
#             self.each_edge_fidelity[(int(s),int(d))] = round(random_fidelity,3)
#             self.each_edge_fidelity[(int(d),int(s))] = round(random_fidelity,3)
#             edge_capacity = round(float(c),3)
            
            
            self.max_edge_capacity = each_edge_capacity_upper_bound
            edge_capacity  = random.uniform(1,each_edge_capacity_upper_bound)
            edge_capacity = float(c)
            self.each_edge_distance[(int(s),int(d))] = float(l)
            self.each_edge_distance[(int(d),int(s))] = float(l)
            self.each_edge_capacity[(int(s),int(d))] = edge_capacity
            self.each_edge_capacity[(int(d),int(s))] = edge_capacity
            self.g.add_edge(int(s),int(d),capacity=edge_capacity,weight=1)
            self.g.add_edge(int(d),int(s),capacity=edge_capacity,weight=1)
        f.close()

        
    def set_nodes_q_value(self):
        for node in self.nodes:
            self.each_node_q_value[node] = self.q_value
    def find_longest_path(self,source,destination):
        # Get the longest path from node source to node destination
        longest_path = max(nx.all_simple_paths(self.g, source, destination), key=lambda x: len(x))
        return longest_path
    def get_required_edge_level_purification_EPR_pairs_all_paths(self,edge,source,destination,wk_idx):
        #print("checking pair %s in edges %s"%(edge,self.set_E))
        if edge not in self.set_E:
            return 1
        else:
            longest_p_lenght = self.find_longest_path(source,destination)
            #print("the length of the longest path is ",len(longest_p_lenght))
            #print("longest_p_lenght",longest_p_lenght)
            return 1
            try:
                if new_target in self.each_edge_target_fidelity[edge]:
                    return self.each_edge_target_fidelity[edge][new_target]
                else:
                    n_avg = self.get_avg_epr_pairs_DEJMPS(edge_basic_fidelity ,new_target)
                    try:
                        self.each_edge_target_fidelity[edge][new_target] = n_avg
                    except:
                        self.each_edge_target_fidelity[edge] = {}
                        self.each_edge_target_fidelity[edge][new_target] = n_avg
                    return n_avg
            except:
                if longest_p_lenght==0:
                    new_target = self.each_edge_fidelity[edge]
                else:
                    new_target = (3*(4/3*max_F_threshold-1/3)**(1/longest_p_lenght)+1)/4

                edge_basic_fidelity = self.each_edge_fidelity[edge]
                n_avg = self.get_avg_epr_pairs_DEJMPS(edge_basic_fidelity ,new_target)
                try:
                    self.each_edge_target_fidelity[edge][new_target] = n_avg
                except:
                    self.each_edge_target_fidelity[edge] ={}
                    self.each_edge_target_fidelity[edge][new_target] = n_avg
            return n_avg    
    def get_path_info(self):
        self.all_user_pairs_across_wks = []
        self.each_pair_paths = {}
        self.each_scheme_each_user_pair_paths = {}
        set_of_all_paths = []
        self.path_counter_id = 0
        self.path_counter = 0
        for wk,ks in self.each_wk_organizations.items():
            for k in ks:
                for u in self.each_wk_each_k_user_pairs[wk][k]:
                    if u not in self.all_user_pairs_across_wks:
                        self.all_user_pairs_across_wks.append(u)
        
        for link_cost_metric in self.link_cost_metrics:
            for user_pair in self.all_user_pairs_across_wks:
                user_pair_id = self.each_pair_id[user_pair]
                self.each_pair_paths[user_pair_id]=[]
        for user_pair in self.all_user_pairs_across_wks:
            having_atleast_one_path_flag = False
            for link_cost_metric in ["EGR","Hop","EGRSquare"]:
                self.each_link_cost_metric =link_cost_metric 
                self.set_link_weight(link_cost_metric)
                user_pair_id = self.each_pair_id[user_pair]
                paths = self.get_paths_between_user_pairs(user_pair)
                path_flag = False
                for path in paths:
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
                    if self.get_basic_fidelity(path_edges)>=0.60:
                        path_flag= True
                        try:
                            self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id].append(self.path_counter_id)
                        except:
                            try:
                                self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=[self.path_counter_id]
                            except:
                                self.each_scheme_each_user_pair_paths[link_cost_metric]={}
                                self.each_scheme_each_user_pair_paths[link_cost_metric][user_pair_id]=[self.path_counter_id]
                        
                        
                        having_atleast_one_path_flag = True
                        set_of_all_paths.append(path_edges)
                        self.set_each_path_length(self.path_counter_id,path)
                        self.set_of_paths[self.path_counter_id] = path_edges
                        try:
                            self.each_pair_paths[user_pair_id].append(self.path_counter_id)
                        except:
                            self.each_pair_paths[user_pair_id] = [self.path_counter_id]
                        self.path_counter_id+=1  
                        self.path_counter+=1
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
    def set_edge_fidelity(self,edge_fidelity_range):
        
        self.max_edge_fidelity = edge_fidelity_range
        self.min_edge_fidelity = edge_fidelity_range
        for edge in self.g.edges:
            random_fidelity = random.uniform(self.min_edge_fidelity,self.max_edge_fidelity)
            self.each_edge_fidelity[edge] = round(random_fidelity,3)
            self.each_edge_fidelity[(int(edge[1]),int(edge[0]))] = round(random_fidelity,3)
            
    def set_link_weight(self,link_cost_metric):
        for edge in self.g.edges:
            edge_capacity = self.each_edge_capacity[edge]
            if link_cost_metric =="Hop":
                weight1=1
                weight2 = 1
                self.g[edge[0]][edge[1]]['weight']=weight1
                self.g[edge[1]][edge[0]]['weight']= weight2
            elif link_cost_metric =="EGR":
                weight1=1/edge_capacity
                weight2 = 1/edge_capacity
                self.g[edge[0]][edge[1]]['weight']=weight1
                self.g[edge[1]][edge[0]]['weight']= weight2
            elif link_cost_metric =="EGRSquare":
                weight1=1/(edge_capacity**2)
                weight2 = 1/(edge_capacity**2)
                self.g[edge[0]][edge[1]]['weight']=weight1
                self.g[edge[1]][edge[0]]['weight']= weight2
            elif link_cost_metric =="Bruteforce":
                weight1=1
                weight2 = 1
                self.g[edge[0]][edge[1]]['weight']=1
                self.g[edge[1]][edge[0]]['weight']= 1
    def update_link_rates(self,alpha_value):
        self.alpha_value = alpha_value
        edge_rates = []
        for edge in self.g.edges:
            edge_length = self.each_edge_distance[edge]
            c = 1
            etha = 10**(-0.1*0.2*edge_length)
            T = (edge_length*10**(-4))/25# based on simulation setup of data link layer paper
            edge_rate = (2*c*etha*alpha_value)/T
            self.g[edge[0]][edge[1]]['capacity']=edge_rate
            self.each_edge_fidelity[edge] = 1-alpha_value
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
        
        num_nodes = len(self.nodes)
        try:
            work_load_file = self.topology_file.split(".txt")[0]
        except:
            if ".txt" not in self.topology_file:
                work_load_file = self.topology_file
        if self.training:
            f = open(work_load_file+"WK2", 'r')
        else:
            f = open(work_load_file+"WK", 'r')
        self.work_load_counter = 0
        all_active_user_pairs_acros_wks = []
        header = f.readline()
        for line in f:
            self.work_loads.append(self.work_load_counter)
            values = line.strip().split(',')#wk_indx,organization,weight,user_pair,weight
            wok_idx = int(values[0])
            k = int(values[1])
            weight = float(values[2])
            i = int(values[3].split(":")[0])
            j = int(values[3].split(":")[1])
            org_weight = float(values[2])
            flow_weight = float(values[4])
            user_pair = (i,j)
            """we check how many flows we have set for each organization"""   
            try:
                num_covered_flows = len(self.each_wk_each_k_user_pairs[wok_idx][k])
            except:
                num_covered_flows = 0
            if num_covered_flows<self.number_of_flows:
                try:
                    self.each_wk_each_k_user_pairs[wok_idx][k].append(user_pair)
                except:
                    self.each_wk_each_k_user_pairs[wok_idx]={}
                    self.each_wk_each_k_user_pairs[wok_idx][k]=[user_pair]

                """we create an id for this flow and added to the data structure
                This is becasue we want to let two organizations have same flows but with different ids"""
                self.each_id_pair[self.pair_id] = user_pair
                self.each_pair_id[user_pair] = self.pair_id
                try:
                    self.each_wk_each_k_user_pair_ids[wok_idx][k].append(self.pair_id)
                except:
                    try:
                        self.each_wk_each_k_user_pair_ids[wok_idx][k]= [self.pair_id]
                    except:
                        self.each_wk_each_k_user_pair_ids[wok_idx]={}
                        self.each_wk_each_k_user_pair_ids[wok_idx][k]= [self.pair_id]
                # we set the weight of each flow
                try:
                    self.each_wk_k_u_weight[wok_idx][k][self.pair_id] = flow_weight
                    self.each_wk_k_u_pair_weight[wok_idx][k][self.pair_id] = flow_weight
                except:
                    try:
                        self.each_wk_k_u_weight[wok_idx][k] ={}
                        self.each_wk_k_u_weight[wok_idx][k][self.pair_id] = flow_weight

                        self.each_wk_k_u_pair_weight[wok_idx][k] ={}
                        self.each_wk_k_u_pair_weight[wok_idx][k][self.pair_id] = flow_weight
                    except:
                        self.each_wk_k_u_weight[wok_idx] ={}
                        self.each_wk_k_u_weight[wok_idx][k] = {}
                        self.each_wk_k_u_weight[wok_idx][k][self.pair_id] = flow_weight

                        self.each_wk_k_u_pair_weight[wok_idx] ={}
                        self.each_wk_k_u_pair_weight[wok_idx][k] ={}
                        self.each_wk_k_u_pair_weight[wok_idx][k][self.pair_id] = flow_weight

                self.pair_id+=1
                """We set the weight of the organization"""
                try:
                    self.each_wk_k_weight[wok_idx][k] = org_weight
                except:
                    self.each_wk_k_weight[wok_idx]= {}
                    self.each_wk_k_weight[wok_idx][k] = org_weight  
                # we set the work load its organization
                try:
                    if k not in self.each_wk_organizations[wok_idx]:
                        try:
                            self.each_wk_organizations[wok_idx].append(k)
                        except:
                            self.each_wk_organizations[wok_idx]=[k]
                except:
                    self.each_wk_organizations[wok_idx]=[k]
                    
    def set_each_k_user_pair_paths(self,wk_idx):
        """we set self.num_of_paths for each user pair of each organization """
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
                
                    if self.get_this_path_fidelity(path_edges)>=0.6:
                        self.set_each_path_length(self.path_counter_id,path)
                        self.set_of_paths[self.path_counter_id] = path_edges
                        path_id=self.path_counter_id 
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
        self.set_each_path_basic_fidelity()
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.set_required_EPR_pairs_for_each_path_each_fidelity_threshold(wk_idx)
        
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
    
    def get_each_wk_k_threshold(self,wk_idx,k):
        return self.each_wk_k_fidelity_threshold[wk_idx][k]

        
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
        self.each_n_f_purification_result = {}
        self.each_edge_target_fidelity = {}
        self.each_u_all_real_paths = {}
        self.each_u_all_real_disjoint_paths = {}
        self.each_u_paths = {}
        self.nodes = []
        self.oracle_for_target_fidelity = {}
        self.each_k_path_path_id = {}
        self.global_each_basic_fidelity_target_fidelity_required_EPRs = {}
        self.all_basic_fidelity_target_thresholds = []
        self.path_counter_id = 0
        self.pair_id = 0
        self.each_u_weight={}
        self.each_path_legth = {}
        self.load_topology()
    
    def set_each_wk_k_fidelity_threshold(self):
        self.each_wk_k_fidelity_threshold = {}
        possible_thresholds_based_on_given_range = []
        
        possible_thresholds_based_on_given_range.append(self.fidelity_threshold_range)
        for wk,ks in self.each_wk_organizations.items():
            for k in ks:
                try:
                    self.each_wk_k_fidelity_threshold[wk][k]= possible_thresholds_based_on_given_range[random.randint(0,len(possible_thresholds_based_on_given_range)-1)]
                except:
                    self.each_wk_k_fidelity_threshold[wk]= {}
                    self.each_wk_k_fidelity_threshold[wk][k] = possible_thresholds_based_on_given_range[random.randint(0,len(possible_thresholds_based_on_given_range)-1)]
    
        
    
    def set_each_path_length(self,path_id,path):
        self.each_path_legth[path_id] = len(path)
    
    def get_next_fidelity_and_succ_prob_BBPSSW(self,F):
        succ_prob = (F+((1-F)/3))**2 + (2*(1-F)/3)**2
        output_fidelity = (F**2 + ((1-F)/3)**2)/succ_prob

        return output_fidelity, succ_prob

    def get_next_fidelity_and_succ_prob_DEJMPS(self,F1,F2,F3,F4):
        succ_prob = (F1+F2)**2 + (F3+F4)**2
        output_fidelity1 = (F1**2 + F2**2)/succ_prob
        output_fidelity2 = (2*F3*F4)/succ_prob
        output_fidelity3 = (F3**2 + F4**2)/succ_prob
        output_fidelity4 = (2*F1*F2)/succ_prob

        return output_fidelity1, output_fidelity2, output_fidelity3, output_fidelity4, succ_prob

    def get_avg_epr_pairs_BBPSSW(self,F_init,F_target):
        F_curr = F_init
        n_avg = 1.0
        while(F_curr < F_target):
            F_curr,succ_prob = get_next_fidelity_and_succ_prob_BBPSSW(F_curr)
            n_avg = n_avg*(2/succ_prob)
        return  n_avg

    def get_avg_epr_pairs_DEJMPS(self,F_init,F_target):
        F_curr = F_init
        F2 = F3 = F4 = (1-F_curr)/3
        n_avg = 1.0
        while(F_curr < F_target):
            F_curr,F2, F3, F4, succ_prob = self.get_next_fidelity_and_succ_prob_DEJMPS(F_curr, F2, F3, F4)
            n_avg = n_avg*(2/succ_prob)
            
        return  n_avg
    
    
    def set_required_EPR_pairs_for_each_path_each_fidelity_threshold(self,wk_idx):
        targets = []
        for k,target_fidelity in self.each_wk_k_fidelity_threshold[wk_idx].items():
            if target_fidelity not in targets:
                targets.append(target_fidelity)
        targets.append(0.6)
        targets.sort()
        counter = 0
        for path,path_basic_fidelity in self.each_path_basic_fidelity.items():
            counter+=1
            try:
                if path_basic_fidelity in self.global_each_basic_fidelity_target_fidelity_required_EPRs:
                    for target in targets:
                        n_avg = self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target]
                        try:
                            self.oracle_for_target_fidelity[path][target] = n_avg
                        except:
                            self.oracle_for_target_fidelity[path] = {}
                            self.oracle_for_target_fidelity[path][target] = n_avg
                else:
                    for target in targets:
                        n_avg = self.get_avg_epr_pairs_DEJMPS(path_basic_fidelity ,target)
                        try:
                            self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg 
                        except:
                            self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity]={}
                            self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg 
                        try:
                            self.oracle_for_target_fidelity[path][target] = n_avg
                        except:
                            self.oracle_for_target_fidelity[path] = {}
                            self.oracle_for_target_fidelity[path][target] = n_avg
            except:
                for target in targets:
                    n_avg  = self.get_avg_epr_pairs_DEJMPS(path_basic_fidelity ,target)
                    try:
                        self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg 
                    except:
                        self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity]={}
                        self.global_each_basic_fidelity_target_fidelity_required_EPRs[path_basic_fidelity][target] =n_avg                     
                    try:
                        self.oracle_for_target_fidelity[path][target] = n_avg
                    except:
                        self.oracle_for_target_fidelity[path] = {}
                        self.oracle_for_target_fidelity[path][target] = n_avg
        #print("this is important ",self.topology_name,self.oracle_for_target_fidelity,self.each_path_basic_fidelity)
    def get_required_edge_level_purification_EPR_pairs(self,edge,p,F,wk_idx):
        return 1
        longest_p_lenght   = 0
        max_F_threshold = 0
        for k,users in self.each_wk_k_user_pairs[wk_idx].items():
            for u in users:
                if p in self.each_k_u_paths[k][u]:
                    path = self.set_of_paths[p]
                    longest_p_lenght = len(path)
                    if self.each_wk_k_fidelity_threshold[wk_idx][k] > max_F_threshold:
                        max_F_threshold = self.each_wk_k_fidelity_threshold[wk_idx][k]
        if longest_p_lenght==0:
            new_target = self.each_edge_fidelity[edge]
            
        else:
            new_target = (3*(4/3*max_F_threshold-1/3)**(1/longest_p_lenght)+1)/4
            
        edge_basic_fidelity = self.each_edge_fidelity[edge]
        try:
            if new_target in self.each_edge_target_fidelity[edge]:
                return self.each_edge_target_fidelity[edge][new_target]
            else:
                n_avg = self.get_avg_epr_pairs_DEJMPS(edge_basic_fidelity ,new_target)
                try:
                    self.each_edge_target_fidelity[edge][new_target] = n_avg
                except:
                    self.each_edge_target_fidelity[edge] = {}
                    self.each_edge_target_fidelity[edge][new_target] = n_avg
                return n_avg
        except:
            if longest_p_lenght==0:
                new_target = self.each_edge_fidelity[edge]
            else:
                new_target = (3*(4/3*max_F_threshold-1/3)**(1/longest_p_lenght)+1)/4
                
            edge_basic_fidelity = self.each_edge_fidelity[edge]
            n_avg = self.get_avg_epr_pairs_DEJMPS(edge_basic_fidelity ,new_target)
            try:
                self.each_edge_target_fidelity[edge][new_target] = n_avg
            except:
                self.each_edge_target_fidelity[edge] ={}
                self.each_edge_target_fidelity[edge][new_target] = n_avg
            return n_avg
    def get_required_purification_EPR_pairs(self,p,threshold):

        return self.oracle_for_target_fidelity[p][threshold]
        
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
                    
      
                    
    def get_basic_fidelity(self,path_edges):
        if path_edges:
            basic_fidelity = 1/4+(3/4)*(4*self.each_edge_fidelity[path_edges[0]]-1)/3
            for edge in path_edges[1:]:
                basic_fidelity  = (basic_fidelity)*((4*self.each_edge_fidelity[edge]-1)/3)
            basic_fidelity = basic_fidelity
        else:
            print("Error")
            return 0.6
        return round(basic_fidelity,3)
    def get_fidelity(self,path_edges):
        if path_edges:
            F_product = (4*self.each_edge_fidelity[path_edges[0]]-1)/3 
            for edge in path_edges[1:]:
                F_product  = F_product*(4*self.each_edge_fidelity[edge]-1)/3

        else:
            print("Error")
            return 0.6
        N = len(path_edges)+1
        p1 = 1
        p2 = 1
        F_final = 1/4*(1+3*(p1*p2)**(N-1)*(F_product))
        return round(F_final,3)
    def set_each_path_basic_fidelity(self):
        self.each_path_basic_fidelity = {}
        for path,path_edges in self.set_of_paths.items():
            if path_edges:
                F_final = self.get_fidelity(path_edges)
                #basic_fidelity = 1/4+(3/4)*(4*self.each_edge_fidelity[path_edges[0]]-1)/3
                #for edge in path_edges[1:]:
                    #basic_fidelity  = (basic_fidelity)*((4*self.each_edge_fidelity[edge]-1)/3)
                #basic_fidelity = basic_fidelity
            else:
                print("Error")
                break
            self.each_path_basic_fidelity[path]= round(F_final,3)

   
    def get_edges(self):
        return self.set_E
    def get_this_path_fidelity(self,path_edges):
        if path_edges:
            F_final = self.get_fidelity(path_edges)
            #basic_fidelity = 1/4+(3/4)*(4*self.each_edge_fidelity[path_edges[0]]-1)/3
            #for edge in path_edges[1:]:
                #basic_fidelity  = (basic_fidelity)*((4*self.each_edge_fidelity[edge]-1)/3)
        else:
            F_final  = 0.999
        return F_final

    def check_path_include_edge(self,edge,path):
        if edge in self.set_of_paths[path] or (edge[1],edge[0]) in self.set_of_paths[path]:
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

    



# In[10]:


# path_edges = [1,2,3,4,5,6,7,8]
# each_edge_fidelity = {1:0.99,2:0.99,3:0.99,4:0.99,5:0.99,6:0.99,7:0.99,8:0.99}
# basic_fidelity = 1/4+(3/4)*(4*each_edge_fidelity[path_edges[0]]-1)/3
# for edge in path_edges[1:]:
#     basic_fidelity  = (basic_fidelity)*((4*each_edge_fidelity[edge]-1)/3)
# print(basic_fidelity)


# In[ ]:





# In[25]:





# In[ ]:





# In[9]:


#parsing_zoo_topologies()



# In[ ]:




