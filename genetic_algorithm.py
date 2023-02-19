#!/usr/bin/env python
# coding: utf-8

# In[3]:


import networkx as nx
from itertools import islice
import random
from itertools import groupby
import time
import math as mt
import csv
import os
import random
import pdb

from os import listdir

from os.path import isfile, join


# In[2]:


class Genetic_algorithm:
    def __init__(self,config):
        self.crossover_p = 1
        self.mutation_p = 1
        self.selection_p = 1
        self.elit_pop_size =10 
        self.multi_point_crossover_value = config.multi_point_crossover_value
        self.multi_point_mutation_value = config.multi_point_mutation_value
        self.number_of_chromosomes = 10# you can set how many chromosome you want to have in each chromosome in config file
        self.max_runs_of_genetic_algorithm = int(config.max_runs_of_genetic_algorithm)# how many generations we want to run genetic algorithm
        self.each_path_replaceable_paths = {}
        self.elit_pop_sizes = config.elit_pop_sizes
        self.cross_over_values=config.cross_over_values
        self.mutation_op_values = config.mutation_op_values
        self.population_sizes = config.population_sizes
        self.random_initial_population = config.genetic_algorithm_random_initial_population
        self.genetic_algorithm_initial_population = config.genetic_algorithm_initial_population
        self.only_elit = config.ga_only_elit_to_generate_population
        self.dynamic_policy_flag = config.dynamic_policy_flag
        self.each_fitness_chromosomes = {}
        self.chromosomes = [[1,2,3],[4,5,6]]# an example of chromosomes. Each list in this list is one chromosome
        #each chromosome is a list of paths ids. For example if we have three user pairs and
        # each user can have at most one path, then in the first chromosom [1,2,3] \n
        # path id 1 is for the first user, 2 is for the second and 3 is for third.
        
    def crossover_op(self,chromosome1,chromosome2):
        """this function applies crossover operator to the given chromosomes"""
        new_chromosome1 = []
        new_chromosome2 = []
        points = []
    
        while (len(points)< self.multi_point_crossover_value):
            random_value = random.randint(0,len(chromosome1)-1)
            if random_value not in points:
                points.append(random_value)
        points.sort()
        start_point = 0
        points_indexes = []
        random_value_crossover = random.uniform(0,1)
        if random_value_crossover <=self.crossover_p:
            flag = True
            for random_point in points:
                if flag:
                    flag = False
                    for i in range(start_point,random_point):
                        new_chromosome2.append(chromosome1[i])
                    for i in range(start_point,random_point):
                        new_chromosome1.append(chromosome2[i])
                else:
                    flag = True
                    for i in range(start_point,random_point):
                        new_chromosome2.append(chromosome2[i])
                    for i in range(start_point,random_point):
                        new_chromosome1.append(chromosome1[i])
                start_point = random_point


            for j in range(max(points),len(chromosome2)):
                    new_chromosome1.append(chromosome2[j])
            for j in range(max(points),len(chromosome2)):
                    new_chromosome2.append(chromosome1[j]) 
        else:
            new_chromosome1= chromosome1
            new_chromosome2 = chromosome2
        assert len(new_chromosome1)==len(new_chromosome2),"Two chromosomes don't have the same length!"
        
        return new_chromosome1,new_chromosome2
    
    def mutation_op(self,chromosome):
        """this function applies mutation operator to the given chromosome"""
        random_value = random.uniform(0,1)
        if random_value <=self.mutation_p:
            points = []
            while(len(points)< self.multi_point_mutation_value):
                random_value = random.randint(0,len(chromosome)-1)
                if random_value not in points:
                    points.append(random_value)
            points.sort()
            for random_point in points:
                possible_values = self.each_path_replaceable_paths[chromosome[random_point]]
                new_value = chromosome[random_point]
                if len(possible_values)>1:
                    while(new_value==chromosome[random_point]):
                        new_value = possible_values[random.randint(0,len(possible_values)-1)]
                chromosome[random_point]= new_value
        return chromosome
    def selection_random_chromosomes(self,number):
        assert number in [1,2],"Either ask one or two chromosomes!"
        if number ==2:
            if len(self.elit_population)==0:
                chromosome1= self.chromosomes[random.randint(0,len(self.chromosomes)-1)]
            elif len(self.elit_population)==1:
                chromosome1= self.elit_population[random.randint(0,len(self.elit_population)-1)]
                chromosome2 = chromosome1
                return chromosome1,chromosome2
            else:
                chromosome1= self.elit_population[random.randint(0,len(self.elit_population)-1)]
                chromosome2 = chromosome1
                while(chromosome2==chromosome1):
                    chromosome2= self.elit_population[random.randint(0,len(self.elit_population)-1)]
               
                return chromosome1,chromosome2
        elif number ==1:
            random_value = random.uniform(0,1)
            if len(self.elit_population)==0:
                chromosome1= self.chromosomes[random.randint(0,len(self.chromosomes)-1)]
            else:
                chromosome1= self.elit_population[random.randint(0,len(self.elit_population)-1)]
            return chromosome1
        
    def select_elit_population(self):
        self.elit_population = []
        # Get Top N fitness values from Records
        N = int(int(self.elit_pop_size/100*len(self.chromosomes))+1)
        top_fitness_values = sorted(list(self.each_fitness_chromosomes.keys()),reverse=True)
        all_fitness_values = list(self.each_fitness_chromosomes.keys())
        for fitness in top_fitness_values:
            for chromosome in self.each_fitness_chromosomes[fitness]:
                if len(self.elit_population)<=N and chromosome not in self.elit_population:
                    self.elit_population.append(chromosome)
        for fitness in top_fitness_values:
            for chromosome in self.each_fitness_chromosomes[fitness]:
                if len(self.elit_population)<=N:
                    self.elit_population.append(chromosome)
        if not self.only_elit:
            while(len(self.elit_population)<self.number_of_chromosomes):
                random_fitnes = all_fitness_values[random.randint(0,len(all_fitness_values)-1)]
                random_chromosome = self.each_fitness_chromosomes[random_fitnes][0]
                counter = 0
                while(random_chromosome in self.elit_population and counter<400):
                    random_fitnes = all_fitness_values[random.randint(0,len(all_fitness_values)-1)]
                    random_chromosome = self.each_fitness_chromosomes[random_fitnes][0]
                    counter+=1
                self.elit_population.append(random_chromosome)
    def population_gen_op(self):
        self.select_elit_population()
    
        new_population = []
        while(len(new_population)<len(self.chromosomes)):
            new_elected_chromosomes = []
            chromosome1,chromosome2 = self.selection_random_chromosomes(2)
            chromosome1,chromosome2 = self.crossover_op(chromosome1,chromosome2)
            new_elected_chromosomes.append(chromosome1)
            new_elected_chromosomes.append(chromosome2)
            chromosome3 = self.selection_random_chromosomes(1)
            chromosome3 = self.mutation_op(chromosome3)
            new_elected_chromosomes.append(chromosome3)
            if new_elected_chromosomes:
                for chromosome in new_elected_chromosomes:
                    new_population.append(chromosome)
        self.chromosomes = new_population
        
    def update_operation_probabilities(self,runs_of_genetic_algorithm,config):
        """this function update the selection, crossover, and mutation probabilities to
        trade-of between exploration and exploitation"""
        pass
        if self.dynamic_policy_flag and runs_of_genetic_algorithm % config.ga_elit_pop_update_step == config.ga_elit_pop_update_step - 1:            
            self.elit_pop_size =max(self.elit_pop_size-5,10) 
            
        if self.dynamic_policy_flag and runs_of_genetic_algorithm % config.ga_crossover_mutation_update_step == config.ga_crossover_mutation_update_step - 1:            

            self.crossover_p = max(self.crossover_p-0.05,0.5)
        if self.dynamic_policy_flag and runs_of_genetic_algorithm % config.ga_crossover_mutation_multi_point_update_step == config.ga_crossover_mutation_multi_point_update_step - 1:
            self.multi_point_crossover_value = max(self.multi_point_crossover_value-1,1)
            self.multi_point_mutation_value = max(self.multi_point_mutation_value-1,1)
        
            #             self.selection_p = min(self.selection_p+0.05,0.4)

#             self.mutation_p = max(self.crossover_p-0.03,0.1)
            #print(" ********* Hyper parameters are updated *************** ")
        
    def generate_chromosomes(self,wk_idx,network):
        """this function generates a population from finite number of chromosomes"""
        network.ordered_user_pairs = []# we use this list to identify the paths of each user pair in the chromosome later
        # we use the data structure filled in function get_each_user_all_paths() for this
        network.valid_flows = []
        for k in network.each_wk_organizations[wk_idx]:
            for user_pair_id in network.each_wk_each_k_user_pair_ids[wk_idx][k]:
                if user_pair_id not in network.valid_flows and len(network.each_user_pair_all_paths[user_pair_id])>0:
                    network.valid_flows.append(user_pair_id)
                network.each_user_organization[user_pair_id] = k
        network.valid_flows.sort()
        self.chromosomes = []
        print("for work load %s we have these flows %s "%(wk_idx,network.valid_flows))
        
        
        # We use the paths computed by EGR scheme to initiate the genetic algorth population
        if self.genetic_algorithm_initial_population=="EGR":
                for i in range(int(self.random_initial_population*self.number_of_chromosomes/100)):
                    chromosome = []# this is a new chromosome
                    for k in network.each_wk_organizations[wk_idx]:# Since we have only one organization, the value of k is always zero
                        for user_pair_id in network.valid_flows:
                            path_counter = 0
                            path_ids = network.each_scheme_each_user_pair_paths["EGR"][user_pair_id]
                            
                            # we want to know what possible values are for this path to be replaced in mutation
                            for p_id1 in path_ids: 
                                possible_values = []
                                for p_id2 in path_ids:
                                    if p_id1 !=p_id2:
                                        possible_values.append(p_id2)
                                self.each_path_replaceable_paths[p_id1] = possible_values
                            for path_id in path_ids:
                                if path_counter <network.num_of_paths:# check how many paths have been selected for each user pair
                                    chromosome.append(path_id)
                                    path_counter+=1
                            for i in range(path_counter,network.num_of_paths):
                                print("path_ids to select one random for flow ",path_ids,user_pair_id)
                                path_id = path_ids[random.randint(0,len(path_ids)-1)]# get one random path
                                chromosome.append(path_id)
                    self.chromosomes.append(chromosome)
        
        
        # if true, we initialize the first population from the results of rl after config.genetic_algorithm_initial_population_rl_epoch_number epoch numbers of training
        elif self.genetic_algorithm_initial_population=="RL":
            print("******* initializing from RL result *********. ")
            with open(network.toplogy_wk_rl_for_initialization_ga_result_file, "r") as f:
                reader = csv.reader( (line.replace('\0','') for line in f) )
                for line in reader:#0:topology_name,1:wk_idx,2:num_of_paths,
#                             3:fidelity_threshold_range,4:q_value,
#                             5:each_link_cost_metric,6:number_of_flows,7:number_of_training_wks,
#                                 8:egr,9:epoch_number,
#                                 10:cut_off_for_path_searching,11:k,12:u,13:p
                    #print("line is ",line)
                    if line and len(line)>5:
                        #print(">5 line is ",line)
                        corrupted_line_flag = True
#                     ['ModifiedSurfnetRL.txt', '4', '3', '0.9', '1', 'Hop', 
#                      '150', '5', '17901.46404501038', '59', '5', '0', '36', '401',
#                      '[(53, 1), (1, 68), (68, 8), (8, 30), (30, 67)]']
                        try:
                            topology_name = line[0]
                            wk_idx = float(line[1])
                            num_of_paths = int(line[2])
                            F = round(float(line[3]),3)
                            q_value = round(float(line[4]),3)
                            num_flows =int(line[6]) 
                            number_of_training_wks = int(line[7])
                            egr = float(line[8])/1000
                            epoch_number = int(line[9])
                            k = int(line[11])
                            flow_id = int(line[12])
                            path_id = int(line[13])
                            corrupted_line_flag = False
                        except ValueError:
                            print(ValueError)
#                             corrupted_line_flag = True
                        if not corrupted_line_flag:
                            if number_of_training_wks == network.number_of_training_wks and  epoch_number==network.genetic_algorithm_initial_population_rl_epoch_number:
                                print("we use the result! that is good")
                                print("we trained rl with %s wks want to use the result of %s wks RL. this the for %s epochs we want to use %s epochs "%(number_of_training_wks,network.number_of_training_wks,epoch_number,network.genetic_algorithm_initial_population_rl_epoch_number))
                                #time.sleep(1)
                                #time.sleep(1)
                                try:
                                    network.each_scheme_each_user_pair_paths["RL"][flow_id].append(path_id)
                                except:
                                    try:
                                        network.each_scheme_each_user_pair_paths["RL"][flow_id]= [path_id]
                                    except:
                                        try:
                                            network.each_scheme_each_user_pair_paths["RL"]= {}
                                            network.each_scheme_each_user_pair_paths["RL"][flow_id]=[path_id]
                                        except ValueError:
                                            print("Error",ValueError)
                        else:
                            print("not a valid line! ",line)
            for i in range(int(self.random_initial_population*self.number_of_chromosomes/100)):
                chromosome = []# this is a new chromosome
                for k in network.each_wk_organizations[wk_idx]:# Since we have only one organization, the value of k is always zero
                    for user_pair_id in network.valid_flows:
                        path_counter = 0
                        try:
                            path_ids = network.each_scheme_each_user_pair_paths["RL"][user_pair_id]
                            print("we have these path ids  %s from RL for flow %s "%(user_pair_id,path_ids))
                            all_path_ids = network.each_user_pair_all_paths[user_pair_id]
#                             print("and all the path ids for this flow are %s "%(all_path_ids))
                            #time.sleep(3)
                        except:
                            try:
                                path_ids = network.each_scheme_each_user_pair_paths["Hop"][user_pair_id]
                            except:
                                try:
                                    path_ids = network.each_scheme_each_user_pair_paths["EGR"][user_pair_id]
                                except:
                                    path_ids =[]
                        # we want to know what possible values are for this path to be replaced in mutation
                        for p_id1 in path_ids: 
                            possible_values = []
                            for p_id2 in path_ids:
                                if p_id1 !=p_id2:
                                    possible_values.append(p_id2)
                            self.each_path_replaceable_paths[p_id1] = possible_values
                        for path_id in path_ids:
                            if path_counter <network.num_of_paths:# check how many paths have been selected for each user pair
                                chromosome.append(path_id)
                                path_counter+=1
                        for i in range(path_counter,network.num_of_paths):
                            path_id = path_ids[random.randint(0,len(path_ids)-1)]# get one random path
                            chromosome.append(path_id)
                self.chromosomes.append(chromosome)
                
                
        # We use the paths computed by EGR scheme to initiate the genetic algorth population
        elif self.genetic_algorithm_initial_population=="Hop":
                for i in range(int(self.random_initial_population*self.number_of_chromosomes/100)):
                    chromosome = []# this is a new chromosome
                    for k in network.each_wk_organizations[wk_idx]:# Since we have only one organization, the value of k is always zero
                        for user_pair_id in network.valid_flows:
                           
                            path_counter = 0
                            path_ids = network.each_scheme_each_user_pair_paths["Hop"][user_pair_id]
                            print("for flow %s we have candidate paths %s "%(user_pair_id,path_ids))
                            # we want to know what possible values are for this path to be replaced in mutation
                            for p_id1 in path_ids: 
                                possible_values = []
                                for p_id2 in path_ids:
                                    if p_id1 !=p_id2:
                                        possible_values.append(p_id2)
                                self.each_path_replaceable_paths[p_id1] = possible_values
                            for path_id in path_ids:
                                if path_counter <network.num_of_paths:# check how many paths have been selected for each user pair
                                    chromosome.append(path_id)
                                    path_counter+=1
                            path_tracher_index=0
                            for i in range(path_counter,network.num_of_paths):
                                
                                
                                path_id = path_ids[path_tracher_index]# get one random path
                                chromosome.append(path_id)
                                path_tracher_index+=1
                                if path_tracher_index==len(path_ids):
                                    path_tracher_index = 0
                    self.chromosomes.append(chromosome)
        print("we initialized %s chromosomes "%(len(self.chromosomes)))
        print("and we will add %s more "%(self.number_of_chromosomes-len(self.chromosomes)))
        time.sleep(5)
        for i in range(self.number_of_chromosomes-int(len(self.chromosomes))):
            chromosome = []# this is a new chromosome
            for k in network.each_wk_organizations[wk_idx]:# Since we have only one organization, the value of k is always zero
                for user_pair_id in network.valid_flows:
                    path_counter = 0
                    while(path_counter <network.num_of_paths):# check how many paths have been selected for each user pair
                        path_ids = network.each_user_pair_all_paths[user_pair_id]
                        path_id = path_ids[random.randint(0,len(path_ids)-1)]# get one random path
                        chromosome.append(path_id)
                        # we want to know what possible values are for this path to be replaced in mutation
                        for p_id1 in path_ids: 
                            possible_values = []
                            for p_id2 in path_ids:
                                if p_id1 !=p_id2:
                                    possible_values.append(p_id2)
                            self.each_path_replaceable_paths[p_id1] = possible_values
                        path_counter+=1
                    #print("for flow %s we have paths %s"%(user_pair_id,path_ids))
                    #time.sleep(1)
            self.chromosomes.append(chromosome)
        
                                
        print("we have %s chromosomes "%(len(self.chromosomes)))
        time.sleep(5)


# In[ ]:





# In[ ]:




