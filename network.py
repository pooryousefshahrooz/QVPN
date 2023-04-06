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
from datetime import datetime
import pdb
from os import listdir
from os.path import isfile, join
from threading import Thread


# In[ ]:





# In[ ]:





# In[ ]:


class Network:
    def __init__(self,config,purification_object ,topology_file,training_flag):
        self.data_dir = './data/'
        self.topology_file = self.data_dir+topology_file
        self.topology_name = topology_file
        self.toplogy_wk_scheme_result  = config.toplogy_wk_scheme_result_file
        self.training = training_flag
        self.set_E = []
        
        self.pair_id = 0
        self.number_of_flows = 1
        
        
        # this is to have minimum rate constraint or not
        self.optimization_problem_with_minimum_rate = config.optimization_problem_with_minimum_rate
       
        self.num_of_paths = int(config.num_of_paths)
        # we use these two metrics to engineer our original topology
        self.target_long_link=1
        self.repeater_placement_distance = 1
        
        # minimum rate for flows
        self.min_flow_rate = config.min_flow_rate
        # we set this flag if we want to store the rates served to each flow
        
        self.get_flow_rates_info = config.get_flow_rates_info
        
        # we have a fixed distilation for the greedy-path-selection schemes
        self.distilation_scheme_for_huristic = config.distilation_scheme_for_huristic
        
        # this is for checking the results of what epoch should be used in genetic algorthm initialization
        self.genetic_algorithm_initial_population_rl_epoch_number =config.genetic_algorithm_initial_population_rl_epoch_number
        
        
        # we use this to get the results of the training with how many training work loads
        self.number_of_training_wks = config.number_of_training_wks
        
        # this is the file that have the results fo rl training after different numbers of training epoch and we use for initialization the genetic algorithm        
        self.toplogy_wk_rl_for_initialization_ga_result_file = config.toplogy_wk_rl_for_initialization_ga_result_file
        self.number_of_flow_set = config.number_of_flow_set
        
        self.each_wk_k_pair_id  ={}
        self.each_wk_k_id_pair ={}
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
        self.each_wk_each_k_each_user_pair_id_all_paths={}
        self.each_wk_each_k_each_user_pair_id_paths = {}
        self.num_of_organizations = int(config.num_of_organizations)
        
        self.each_k_path_path_id = {}
        self.each_wk_each_k_user_pairs = {}
        self.each_wk_each_k_user_pair_ids = {}
        self.each_user_pair_all_paths = {}
        self.each_k_weight = {}
        self.each_k_u_weight = {}
        self.each_wk_k_u_pair_weight = {}
        self.each_pair_id_paths = {}
        self.each_scheme_each_user_pair_id_paths = {}
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
        
        self.candidate_paths_size_for_genetic_alg = config.candidate_paths_size_for_genetic_alg
        #we use the object of purification for taking of purificaiton and fidelity parameters
        self.purification = purification_object
        self.purification.set_of_edge_level_Fth = config.set_of_edge_level_Fth
        
        
        
        # we keep track of virtual links in order to set their weights to zero in path computation
        self.set_of_virtual_links = []
        
        # we need this to set the flow of each path. we need the flow Fth and the workload id and k id of a pair
        self.each_user_pair_id_work_load = {}
        self.each_user_pair_id_organization = {}
        # we load the topology 
        self.load_topology()
        
    def split(self,x, n):
        points = []
        
        
        # If x % n == 0 then the minimum
        # difference is 0 and all
        # numbers are x / n
        if (x % n == 0):
            for i in range(n):
    #             print(x//n, end =" ")
                points.append(x//n)
        else:
            # upto n-(x % n) the values
            # will be x / n
            # after that the values
            # will be x / n + 1
            zp = n - (x % n)
            pp = x//n
            for i in range(n):
                if(i>= zp):
    #                 print(pp + 1, end =" ")
                    points.append(pp + 1)
                else:
    #                 print(pp, end =" ")
                    points.append(pp)
        return points
       



    def engineer_topology(self,target_original_network,target_link_lenght,dividing_link_metric):
        onlyfiles = [f for f in listdir("data/") if isfile(join("data/", f))]
        not_connected = 0
        connected = 0
        target_original_network = target_original_network.split("Modified")[0]
        for file in onlyfiles:
            # computing distance between two nodes in graphgml data structure
            #print("this is the file ",file)
            if "graphml" in file and (target_original_network in file and "Modified" not in file):
                G = nx.read_graphml("data/"+file)
                if nx.is_connected(G):
                    lines = ["Link_index	Source	Destination	Capacity(EPRps)	Length(km)	Real/Virtual"+"\n"]
                    randomly_selected_nodes = []
                    each_real_link_lenght = {}
                    each_real_link_rate = {}
                    real_links_rates = []
                    connected+=1
                    latitude_values = []
                    longitude_values = []
                    degrees = []
                    covered_edges = []
                    diameter = nx.diameter(G)

                    distances = []
                    latitudes = nx.get_node_attributes(G,"Latitude")

                    Longitudes  = nx.get_node_attributes(G,"Longitude")
        #             if file =="Darkstrand.graphml":
        #                 print("Longitudes",Longitudes)
        #                 print("latitudes",latitudes)
                    counter = 0
                    import haversine as hs
                    nodes = []
                    for node in G.nodes:

                        nodes.append(int(node))
                        #print(node)
                        degrees.append(G.degree[node])
                        try:
                            #print("node %s latitudes %s Longitudes %s"%(node, latitudes[node],Longitudes[node]))
                            longitude_values.append(Longitudes[node])
                            latitude_values.append(latitudes[node])
                        except:
                            counter+=1
                            pass
                    computed_distances = []
                    for edge in G.edges:
                        flag = True
                        try:
                            coords_1 = (latitudes[edge[0]],Longitudes[edge[0]])
                        except:
                            flag = False
                        try:
                            coords_2 = (latitudes[edge[1]],Longitudes[edge[1]])
                        except:
                            flag = False
                        if flag:
                            distance = hs.haversine(coords_1,coords_2)
                            computed_distances.append(distance)
                    edge_indx = 0
                    node_max_id = max(nodes)+1
                    real_edges = []
                    all_distances = []
                    new_repeaters_id = max(nodes)+1
                    for edge in G.edges:
                        flag = True
                        try:
                            coords_1 = (latitudes[edge[0]],Longitudes[edge[0]])
                        except:
                            flag = False
                        try:
                            coords_2 = (latitudes[edge[1]],Longitudes[edge[1]])
                        except:
                            flag = False
                        if flag:
                            distance = hs.haversine(coords_1,coords_2)
                            if distance ==0:
                                distance = 1
    #                         else:
    #                             print("one reasonable result")
                        else:
                            distance = round(random.uniform(min(computed_distances),max(computed_distances)),3)
                            #print("a randomly generated among %s %s %s"%(min(computed_distances),max(computed_distances),distance))
                        distance = round(distance,3)
                        if int(distance)<=target_link_lenght:
                            new_distance=distance
                            c = 1
                            etha = 10**(-0.1*0.2*new_distance)
                            T = (new_distance*10**(-4))/25# based on simulation setup of data link layer paper
                            if T==0:
                                T=1 * 10**(-6)
                            alpha = 1
                            all_distances.append(new_distance)
                            edge_rate = round((2*c*etha*alpha)/T,3)
                            real_links_rates.append(edge_rate)
                            each_real_link_rate[(int(edge[0]),int(edge[1]))] = edge_rate
                            each_real_link_rate[(int(edge[1]),int(edge[0]))] = edge_rate
                            each_real_link_lenght[(int(edge[0]),int(edge[1]))] = new_distance
                            each_real_link_lenght[(int(edge[1]),int(edge[0]))] = new_distance
                            real_edges.append((int(edge[0]),int(edge[1])))

                            line = str(edge_indx)+"\t"+str(edge[0])+"\t"+str(edge[1])+"\t"+str(edge_rate)+"\t"+str(new_distance)+"\t"+"real"+"\n"
                            edge_indx+=1
                            lines.append(line)
                        else:

                            n = int(math.ceil(distance/target_link_lenght))
                            list_points = split(distance, n)
                            flag = False
                            source = int(edge[0])
                            end = new_repeaters_id
                            flag = True

                            for indx,point in enumerate(list_points):
                                new_distance = point
                                if flag:
                                    flag = False
                                    source = int(edge[0])
                                    end = new_repeaters_id
                                    new_repeaters_id+=1
                                else:
                                    if indx== len(list_points)-1:
                                        source = end
                                        end = int(edge[1])
                                        new_repeaters_id+=1
                                    else:
                                        source = end
                                        end = new_repeaters_id
                                        new_repeaters_id+=1

                                c = 1
                                etha = 10**(-0.1*0.2*new_distance)
                                T = (new_distance*10**(-4))/25# based on simulation setup of data link layer paper
                                if T==0:
                                    T=1 * 10**(-6)
                                alpha = 1
                                all_distances.append(new_distance)
                                edge_rate = round((2*c*etha*alpha)/T,3)
                                real_links_rates.append(edge_rate)
                                each_real_link_rate[(source,end)] = edge_rate
                                each_real_link_rate[(end,source)] = edge_rate
                                each_real_link_lenght[(source,end)] = new_distance
                                each_real_link_lenght[(end,source)] = new_distance
                                real_edges.append((source,end))
                                line = str(edge_indx)+"\t"+str(source)+"\t"+str(end)+"\t"+str(edge_rate)+"\t"+str(new_distance)+"\t"+"real"+"\n"
                                edge_indx+=1
                                lines.append(line)





                    covered_edges = []
                    added_edges = 0
                    node_max_id = new_repeaters_id+1
                    for edge in real_edges:
                        range_distance = max(all_distances) 
                        edge_distance = each_real_link_lenght[edge]
                        paralel_links = 3
                        for i in range(paralel_links):
    #                         distance = round(random.uniform(edge_distance/2,3/2 *edge_distance),2)
                            distance= edge_distance
                            node1=edge[0]
                            node2=edge[1]
                            new_node1 = node_max_id
                            added_edges+=1
                            node_max_id = node_max_id+1
                            covered_edges.append((node1,node2))
                            edge_rate = max(real_links_rates)+100
                            distance = 0
                            line = str(edge_indx)+"\t"+str(node1)+"\t"+str(new_node1)+"\t"+str(edge_rate)+"\t"+str(distance)+"\t"+"virtual"+"\n"
                            edge_indx+=1
                            lines.append(line)

                            #distance = round(random.uniform(min(all_distances),max(all_distances)),2)
                            distance=edge_distance
                            edge_rate = each_real_link_rate[edge]
                            line = str(edge_indx)+"\t"+str(new_node1)+"\t"+str(node2)+"\t"+str(edge_rate)+"\t"+str(distance)+"\t"+"real"+"\n"
                            edge_indx+=1
                            lines.append(line)



                         # Open the file in append & read mode ('a+')
                    file_name = file.split(".graphml")[0]
                    if os.path.exists("data/Modified"+target_original_network+".txt"):
                        os.remove("data/Modified"+target_original_network+".txt")
                    with open("data/Modified"+target_original_network+".txt", "a+") as file_object:
                        for line in lines:
                            file_object.write(line)
                else:
                    not_connected+=1
        
        
    def evaluate_rl_for_path_selection(self,config):
        """this function implements the main work flow of the reinforcement learning algorithm"""
        # input datetime
        dt = datetime(2018, 10, 22, 0, 0)
        # epoch time
        epoch_time = datetime(1970, 1, 1)

        # subtract Datetime from epoch datetime
        delta = (dt - epoch_time)
        rl = RL(config)
        solver = Solver()
        self.each_link_cost_metric ="Hop" 
        self.set_link_weight("Hop")
        Thread(target = rl.train, args=(config,self,)).start()
        time.sleep(10)
#         Thread(target = rl.test, args=(config,self,)).start()

                
    def evaluate_shortest_path_routing(self,config,link_cost_metric):
        """this function evaluates the entanglement generation rate using 
        shortest paths computed based on the given link cost metric"""
        
        
        self.each_wk_each_k_each_user_pair_id_all_paths={}
        self.each_wk_each_k_each_user_pair_id_paths = {}
        solver = Solver()
        self.each_link_cost_metric =link_cost_metric 
        self.set_link_weight(link_cost_metric)
        workloads_with_feasible_solution = []
        purification_schemes_string = ""
        all_work_loads_egr=[]
        for wk_idx in self.work_loads:
            time_in_seconds = time.time()
            self.set_paths_in_the_network(wk_idx)
            """we set the required EPR pairs to achieve each fidelity threshold"""
            self.purification.set_required_EPR_pairs_for_distilation(wk_idx,self)
            # calling the IBM CPLEX solver to solve the optimization problem
            
            egr = solver.CPLEX_maximizing_EGR(wk_idx,self,0,0)
            egr = round(egr,3)
#             egr = solver.CPLEX_maximizing_EGR(wk_idx,self,0,0)
#             egr = solver.CPLEX_maximizing_delta(wk_idx,self,0,0)
#             egr = solver.CPLEX_maximizing_EGR_limited_rate_on_paths(wk_idx,self,0,0)
            if egr >0:
                workloads_with_feasible_solution.append(wk_idx)
            all_work_loads_egr.append(egr)
            #network_capacity = solver.CPLEX_swap_scheduling(wk_idx,self)
            network_capacity= None
#             print("for top %s work_load %s min_rate %s alpha_value %s metric %s number of paths %s we have egr as %s "%
#                   (self.topology_name,wk_idx,min_rate,self.alpha_value,link_cost_metric,self.num_of_paths,egr))
            print("for top %s wk_idx %s alpha %s p2=%s eta=%s min_R= %s metric %s |U_k|=%s #paths %s egr= %s "%
                      (self.topology_name,wk_idx,self.alpha_value,self.purification.two_qubit_gate_fidelity,
                       self.purification.measurement_fidelity,
                       self.min_flow_rate,link_cost_metric,self.number_of_flows,
                       self.num_of_paths,egr))
            self.save_results(wk_idx,config,False,False,True,None,0,egr,0,0,time_in_seconds)
        
            

#             with open(config.toplogy_wk_setup_feasibility_checking, 'a') as newFile:                                
#                 newFileWriter = csv.writer(newFile)
#                 newFileWriter.writerow([self.topology_name,
#                 wk_idx,self.alpha_value,self.purification.two_qubit_gate_fidelity,
#                 self.purification.measurement_fidelity,self.num_of_paths,egr,
#                 self.target_long_link,self.repeater_placement_distance,
#                 self.min_flow_rate,purification_schemes_string,self.number_of_flows])
#         print("workloads_with_feasible_solution ",workloads_with_feasible_solution)
#         print("average egr ",(sum(all_work_loads_egr)/len(all_work_loads_egr)))
#         Edge_Fth = self.purification.set_of_edge_level_Fth[0]
#         with open("results/feasibility_checkingv4.csv", 'a') as newFile:                                
#                 newFileWriter = csv.writer(newFile)
#                 newFileWriter.writerow([self.topology_name,self.alpha_value,
#                 self.purification.two_qubit_gate_fidelity,
#                 self.purification.measurement_fidelity
#                 ,self.number_of_flows,Edge_Fth,len(workloads_with_feasible_solution),sum(all_work_loads_egr)/len(all_work_loads_egr)])
        
        
    def get_each_user_all_paths(self,wk_idx,rl_testing_flag):
        """this function will set all the paths of each user pair for the given work load"""
        
#         if rl_testing_flag:
#             print("*********************** getting all user pairs all paths of workload %s ***************"%(wk_idx))
        self.each_user_pair_id_all_paths = {}
        if rl_testing_flag:
            """this part is for the user pairs that exist only in the testing workload but are not in trainign workload"""
            for k, user_pair_ids in self.each_testing_wk_each_k_user_pair_ids[wk_idx].items():
                #print("we have these user pairs %s in work load %s "%(user_pair_ids,wk_idx))
                for user_pair_id in user_pair_ids:
                    user_pair = self.each_wk_k_id_pair[wk_idx][k][user_pair_id]
                    self.each_user_pair_id_all_paths[user_pair_id]= []
                    for path_id in self.each_pair_id_paths[user_pair_id]:
                        try:
                            self.each_user_pair_id_all_paths[user_pair_id].append(path_id)
                        except:
                            self.each_user_pair_id_all_paths[user_pair_id] = [path_id]
        else:
            for k, user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
                for user_pair_id in user_pair_ids:
                    user_pair = self.each_wk_k_id_pair[wk_idx][k][user_pair_id]
                    self.each_user_pair_id_all_paths[user_pair_id]= []
                    for path_id in self.each_pair_id_paths[user_pair_id]:
                        try:
                            self.each_user_pair_id_all_paths[user_pair_id].append(path_id)
                        except:
                            self.each_user_pair_id_all_paths[user_pair_id] = [path_id]
                        

            
                        
                        
                        
    def set_paths_from_chromosome(self,wk_idx,chromosome):
        """this function uses the information in the chromosome 
        to set the paths to the data structure that will be used by solver"""
        

        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.purification.set_required_EPR_pairs_for_distilation(wk_idx,self)
        
        
        def check_any_version_of_the_path_has_been_added(path_id,path_ids,path_versions):
            for path in path_versions:
                if path in path_ids:
                    return False
            return True
        path_indx = 0
        for k in self.each_wk_organizations[wk_idx]:
            for user_pair_id in self.each_wk_each_k_user_pair_ids[wk_idx][k]:
                path_ids = chromosome[path_indx:path_indx+self.num_of_paths]
                path_ids = list(set(path_ids))

#                     print("for wk %s k %s u %s we have %s paths "%(wk_idx,k,user_pair_id,len(path_ids)))
#                     for p in path_ids:
#                         print("versions: ",p,self.purification.each_path_version_numbers[p])
#                         print("g_value:  ",p,self.purification.each_path_edge_level_g[p])
                # all of the 20 lines of code is to avoid having a duplicate path even a path with duplicate distilation strategy
                # we pick th path with highest value of g function for edge-level distilation 
                new_path_ids = []
                each_path_versions ={}
                for path_id in path_ids:
                    each_path_versions[path_id] = [path_id]
                    if path_id!=-1:
                        for sec_v_path_id in path_ids:
                            if path_id!= sec_v_path_id:
                                if sec_v_path_id in self.purification.each_path_version_numbers[path_id]:
                                    try:
                                        each_path_versions[path_id].append(sec_v_path_id)
                                    except:
                                        each_path_versions[path_id]=[sec_v_path_id]
                for path_id,version_path_ids in each_path_versions.items():
                    edge_g_value = self.purification.each_path_edge_level_g[path_id]
                    edge_level_g_values = []
                    for path_id_v in version_path_ids:
                        edge_level_g_values.append(self.purification.each_path_edge_level_g[path_id_v])
                    if edge_g_value >= max(edge_level_g_values):
                        if path_id not in new_path_ids:
                            if check_any_version_of_the_path_has_been_added(path_id,new_path_ids,self.purification.each_path_version_numbers[path_id]):
                                if self.purification.check_path_distilability(path_id):
                                    new_path_ids.append(path_id)

                path_ids = new_path_ids
                path_indx = path_indx+self.num_of_paths
                k = self.each_user_organization[user_pair_id]
#                     for path_id in path_ids:
#                         for path_id2 in path_ids:
#                             if path_id1!=path_id2 and both_same():
#                     print("for wk %s k %s u %s we are adding %s paths "%(wk_idx,k,user_pair_id,len(path_ids)))
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
        
        
        
    def save_rl_results_for_genetic_initialization(self,config,wk_idx,epoch_number,egr):
        """we save the paths that the rl suggests for this work load and use ot later 
        to initialize the genetic algorthm first population"""
        with open(self.toplogy_wk_rl_for_initialization_ga_result_file, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                for k in self.each_wk_organizations[wk_idx]:
                    for u in self.each_wk_each_k_user_pair_ids[wk_idx][k]: 
                        for p in self.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]:
                            newFileWriter.writerow([self.topology_name,wk_idx,self.num_of_paths,
                            0,self.q_value,
                            self.each_link_cost_metric,self.number_of_flows,config.number_of_training_wks,
                                egr,epoch_number,
                                config.cut_off_for_path_searching,k,u,p,self.set_of_paths[p]])
        
        
    def save_results(self,wk_idx,config,genetic_alg_flag,rl_flag,shortest_path_flag,genetic_alg,
                     runs_of_algorithm,egr,optimal_egr,run_number,time_in_seconds):
        
        purification_schemes_string = ""
        for pur_scheme in self.purification.allowed_purification_schemes:
            if purification_schemes_string:
                purification_schemes_string = purification_schemes_string+","+pur_scheme
            else:
                purification_schemes_string = pur_scheme
        if genetic_alg_flag:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                0,purification_schemes_string,self.q_value,
                "Genetic",self.number_of_flows,genetic_alg.elit_pop_size,
                                        genetic_alg.crossover_p,genetic_alg.mutation_p,runs_of_algorithm,
                                        egr,genetic_alg.number_of_chromosomes,run_number,
                                        config.genetic_algorithm_random_initial_population,
                                        config.ga_elit_pop_update_step,config.ga_crossover_mutation_update_step,
                                        config.cut_off_for_path_searching,config.multi_point_mutation_value,
                                        config.multi_point_crossover_value,
                                        config.ga_crossover_mutation_multi_point_update_step,
                                        config.genetic_algorithm_initial_population_rl_epoch_number,
                                       config.number_of_training_wks,
                                        config.genetic_algorithm_initial_population,
                                        self.purification.two_qubit_gate_fidelity,
                                        self.purification.measurement_fidelity,time_in_seconds,
                                        self.candidate_paths_size_for_genetic_alg,len(config.set_of_edge_level_Fth)])
        elif shortest_path_flag:
            Edge_Fth = self.purification.set_of_edge_level_Fth[0]
            scheme = self.each_link_cost_metric+"-Edge_Fth="+str(Edge_Fth)
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,self.alpha_value,self.num_of_paths,
                0,purification_schemes_string,self.q_value,
                scheme,self.number_of_flows,0,0,
                                        0,1,
                                        egr,0,run_number,
                                        0,
                                        0,0,
                                        config.cut_off_for_path_searching,0,
                                        0,0,self.purification.two_qubit_gate_fidelity,
                                        self.purification.measurement_fidelity,time_in_seconds])
        elif rl_flag:
            with open(self.toplogy_wk_scheme_result, 'a') as newFile:                                
                newFileWriter = csv.writer(newFile)
                newFileWriter.writerow([self.topology_name,wk_idx,
                                        self.alpha_value,self.num_of_paths,
                0,purification_schemes_string,self.q_value,
                "RL",self.number_of_flows,0,0,
                                        0,runs_of_algorithm,
                                        egr,0,run_number,
                                        0,
                                        0,0,
                                        config.cut_off_for_path_searching,0,
                                        0,0,config.rl_batch_size,config.initial_learning_rate,
                                       config.learning_rate_decay_rate,
                                       config.moving_average_decay,
                                        config.learning_rate_decay_step_multiplier,
                                        config.learning_rate_decay_step,
                                       config.entropy_weight,config.optimizer,config.scale,
                                        config.max_step,config.number_of_training_wks,
                                        self.purification.two_qubit_gate_fidelity,
                                        self.purification.measurement_fidelity,time_in_seconds,
                                       len(config.set_of_edge_level_Fth)])
        
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
            if wk_idx in config.workloads_with_feasible_solution:
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
                                        # input datetime
                                        time_in_seconds = time.time()
                                        for chromosome in genetic_alg.chromosomes:
                                            self.set_paths_from_chromosome(wk_idx,chromosome)
                                            fitness_value  = solver.CPLEX_maximizing_EGR(wk_idx,self,runs_of_genetic_algorithm,chromosome_id)
                                            fitness_value = round(fitness_value,3)
                                            try:
                                                genetic_alg.each_fitness_chromosomes[fitness_value].append(chromosome)
                                            except:
                                                genetic_alg.each_fitness_chromosomes[fitness_value] = [chromosome]

                                            # we save the max egr at this point of genetic algorithm
                                            self.save_results(wk_idx,config,True,False,False,genetic_alg,runs_of_genetic_algorithm,fitness_value,0,i,time_in_seconds)
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
    
    
   
                                    
    def load_topology(self):
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
            real_virtual = "real"
            try:
                i, s, d,  c,l,real_virtual = link
            except:
                i, s, d,  c,l = link
            if real_virtual =="virtual":
                self.set_of_virtual_links.append((int(s),int(d)))
                self.set_of_virtual_links.append((int(d),int(s)))
                
                self.purification.virtual_links.append((int(s),int(d)))
                self.purification.virtual_links.append((int(d),int(s)))
                
            if int(s) not in self.nodes:
                self.nodes.append(int(s))
            if int(d) not in self.nodes:
                self.nodes.append(int(d))
            self.set_E.append((int(s),int(d)))
            edge_capacity = float(c)
#             edge_capacity = 10000
            if edge_capacity >self.max_edge_capacity:
                self.max_edge_capacity = edge_capacity
                self.purification.upper_bound_on_distilation = self.max_edge_capacity+100
            
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
        self.each_pair_id_paths = {}
        self.each_scheme_each_user_pair_id_paths = {}
        set_of_all_paths = []
        self.path_counter_id = 0
      
#         print("self.each_wk_organizations",self.each_wk_organizations)
#         for wk,ks in self.each_testing_wk_organizations.items():
#             for k in ks:
#                 try:
#                     for u in self.each_testing_wk_each_k_user_pairs[wk][k]:
#                         if u not in self.all_user_pairs_across_wks:
#                             self.all_user_pairs_across_wks.append(u)
#                 except:
#                     pass
                
#         for wk,ks in self.each_wk_organizations.items():
#             for k in ks:
# #                 print("these are ks ",ks)
# #                 print("and these are what we have set pairs for them ",self.each_wk_each_k_user_pairs[wk])
#                 for u in self.each_wk_each_k_user_pairs[wk][k]:
#                     if u not in self.all_user_pairs_across_wks:
#                         #if self.running_path_selection_scheme=="RL":
#                         self.all_user_pairs_across_wks.append(u)

        for wk_idx,ks in self.each_wk_organizations.items():
            print("extracting paths info for wk %s out of %s "%(wk_idx,len(list(self.each_wk_organizations.keys()))))
            for k in ks:
                for user_pair in self.each_wk_each_k_user_pairs[wk_idx][k]:
                #for user_pair in self.all_user_pairs_across_wks:
                    having_atleast_one_path_flag = False
                    for link_cost_metric in ["Hop","EGR","EGRSquare"]:
        #             for link_cost_metric in ["Hop"]:#just for now to fix the isuue with iniializing the genetic algorthm
                        self.each_link_cost_metric =link_cost_metric 
                        self.set_link_weight(link_cost_metric)
        #                 k = self.each_user_organization[user_pair]
                        user_pair_id = self.each_wk_k_pair_id[wk_idx][k][user_pair]
                        paths = self.get_paths_between_user_pairs(user_pair)
                        random.shuffle(paths)
                        selected_path_for_this_scheme = []
                        path_flag = False
                        for path in paths:
                            node_indx = 0
                            path_edges = []
                            for node_indx in range(len(path)-1):                
                                path_edges.append((path[node_indx],path[node_indx+1]))
                                node_indx+=1

                            #if path_edges not in set_of_all_paths:
                            path_fidelity  = self.purification.get_fidelity(path_edges,self.set_of_virtual_links)
                            #if path_fidelity>0.50:
        #                     wk_idx=self.each_user_pair_id_work_load[user_pair_id]
        #                     k = self.each_user_pair_id_organization[user_pair_id]

                            path_flag= True
                            having_atleast_one_path_flag = True
                            set_of_all_paths.append(path_edges)
                            set_of_different_version_of_a_path = []
                            for edge_level_Fth in self.purification.set_of_edge_level_Fth:
                                
                                self.purification.each_path_edge_level_Fth[self.path_counter_id]=edge_level_Fth

                                self.set_each_path_length(self.path_counter_id,path_edges)
                                self.set_of_paths[self.path_counter_id] = path_edges
                                # we set what is the fidelity threshold of the flow that uses this path
                                self.purification.each_path_basic_fidelity[self.path_counter_id]= round(path_fidelity,3)
                                #print("these are the user pair ids in wk %s k %s user_pair %s user_pair_id %s: %s"%(wk_idx,k,user_pair,user_pair_id,self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k]))
#                                 print("we set Fth %s for wk %s k %s path %s from flow %s "%(self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id],wk_idx,k,self.path_counter_id,user_pair_id))
                                self.purification.each_path_flow_target_fidelity[self.path_counter_id] = self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id]
                                if self.path_counter_id not in selected_path_for_this_scheme:
                                    selected_path_for_this_scheme.append(self.path_counter_id)
                                    
#                                 print("we set path wk %s k %s flow %s with id %s path id %s "%(wk_idx,k,user_pair,user_pair_id,self.path_counter_id))
                                try:
                                    self.each_pair_id_paths[user_pair_id].append(self.path_counter_id)
                                except:
                                    self.each_pair_id_paths[user_pair_id] = [self.path_counter_id]
                                """here we set what is the purification scheme for each path id.
                                for each path, there could multiple versions
                                depends on the number of purification schemes
                                e.g, end-level,edge-level, multi-hop level"""
                                #self.purification.each_path_id_purificaiton_scheme[self.path_counter_id] = pur_scheme
                                set_of_different_version_of_a_path.append(self.path_counter_id)
                                self.path_counter_id+=1  

                            #we put different versions of a path in order to prevent using a path with differernt purification schemes at the same time
                            for path_id in set_of_different_version_of_a_path:
                                self.purification.each_path_version_numbers[path_id] = set_of_different_version_of_a_path


                        #if link_cost_metric=="Hop":
                            #print("scheme %s flow %s have these path ids %s "%(link_cost_metric,user_pair_id,selected_path_for_this_scheme))
                            #time.sleep(1)
                        try:
                            self.each_scheme_each_user_pair_id_paths[link_cost_metric][user_pair_id]=selected_path_for_this_scheme
                        except:
                            self.each_scheme_each_user_pair_id_paths[link_cost_metric]={}
                            self.each_scheme_each_user_pair_id_paths[link_cost_metric][user_pair_id] = selected_path_for_this_scheme

                        if not path_flag:# the flow does not have a valid path in this scsheme (path with fidelity higher than 0.6)
                            try:
                                self.each_scheme_each_user_pair_id_paths[link_cost_metric][user_pair_id]=[]
                            except:
                                self.each_scheme_each_user_pair_id_paths[link_cost_metric]={}
                                self.each_scheme_each_user_pair_id_paths[link_cost_metric][user_pair_id] = []
                    if not having_atleast_one_path_flag:#the flow does not have a valid path in this scsheme (path with fidelity higher than 0.6 using any scheme)
                        self.each_pair_id_paths[user_pair_id] = []
#             print("done!")
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
#             edge_rate = 2000
            self.g[edge[0]][edge[1]]['capacity']=edge_rate
            self.each_edge_capacity[(edge[0],edge[1])] = edge_rate
            self.each_edge_capacity[(edge[1],edge[0])] = edge_rate
            self.purification.each_edge_fidelity[edge] = edge_fidelity
            self.purification.each_edge_fidelity[(edge[1],edge[0])] = edge_fidelity
            edge_rates.append(edge_rate)
            #print("alpha %s generated %s with fidelity %s "%(alpha_value,edge_rate,1-alpha_value))
            
        self.max_edge_capacity = max(edge_rates)
        self.purification.upper_bound_on_distilation = self.max_edge_capacity+100
        
    def set_flows_of_organizations(self):
        """This function reads the workload from the topology+WK file and set the data structures"""
        self.each_wk_k_weight = {}
        self.each_wk_organizations={}
        self.each_wk_each_k_user_pair_ids = {}
        self.each_wk_each_k_user_pairs = {}
        self.pair_id = 0
        self.each_wk_k_id_pair ={}
        self.each_wk_k_pair_id={}
        self.work_loads=[]
        self.each_wk_k_u_weight ={}
        self.each_wk_k_u_pair_weight = {}
        self.each_user_pair_id_work_load = {}
        self.each_user_pair_id_organization = {}
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
#         if self.running_path_selection_scheme in ["EGR","EGRSquare","Hop","Genetic"]:
#             f = open(work_load_file+"WK2", 'r')
#         else:
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
                        self.each_user_pair_id_work_load[user_pair_id] = wk_idx
                        self.each_user_pair_id_organization[user_pair_id] = k
                        try:
                            self.each_wk_each_k_user_pairs[wk_idx][k].append(user_pair)
                        except:
                            try:
                                self.each_wk_each_k_user_pairs[wk_idx][k]= [user_pair]
                            except:
                                self.each_wk_each_k_user_pairs[wk_idx]={}
                                self.each_wk_each_k_user_pairs[wk_idx][k]=[user_pair]
                        """we create an id for this flow and added to the data structure
                        This is becasue we want to let two organizations have same flows but with different ids"""
                        try:
                            self.each_wk_k_id_pair[wk_idx][k][user_pair_id] = user_pair
                        except:
                            try:
                                self.each_wk_k_id_pair[wk_idx][k]={}
                                self.each_wk_k_id_pair[wk_idx][k][user_pair_id] = user_pair
                            except:
                                self.each_wk_k_id_pair[wk_idx]={}
                                self.each_wk_k_id_pair[wk_idx][k] = {} 
                                self.each_wk_k_id_pair[wk_idx][k][user_pair_id] = user_pair
                                
                        try:
                            self.each_wk_k_pair_id[wk_idx][k][user_pair] = user_pair_id
                        except:
                            try:
                                self.each_wk_k_pair_id[wk_idx][k]={}
                                self.each_wk_k_pair_id[wk_idx][k][user_pair] = user_pair_id
                            except:
                                self.each_wk_k_pair_id[wk_idx]={}
                                self.each_wk_k_pair_id[wk_idx][k] ={}
                                self.each_wk_k_pair_id[wk_idx][k][user_pair] = user_pair_id
                            
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
        
#         if self.running_path_selection_scheme in ["RL"]:
#             f = open(work_load_file+"WK2", 'r')
#             num_covered_flows = 0
#             try:
#                 work_load_file = self.topology_file.split(".txt")[0]
#             except:
#                 if ".txt" not in self.topology_file:
#                     work_load_file = self.topology_file

#             header = f.readline()
# #             print("we are testing RL and we are adding testing workloads ",self.workloads_to_test)
#             for line in f:
#                 values = line.strip().split(',')#wk_indx,organization,weight,user_pair,weight
#                 wk_idx = int(values[0])
#                 k = int(values[1])
#                 if wk_idx<self.workloads_to_test and k < self.num_of_organizations:
#                     org_weight = float(values[2])
#                     organization_F = float(values[3])
#                     i = int(values[4].split(":")[0])
#                     j = int(values[4].split(":")[1])
#                     flow_weight = float(values[5])
#                     flow_fidelity_threshold = float(values[6])
#                     user_pair = (i,j)
#                     flow_number_set = int(values[7])
#                     try:
#                         user_pair_id = self.each_pair_id[user_pair] 
#                     except:
#                         user_pair_id = self.pair_id
#                         self.pair_id+=1
                    
#                     if flow_number_set ==self.number_of_flows:
#                         if wk_idx not in self.testing_work_loads:
#                             self.testing_work_loads.append(wk_idx)

#                         """we check how many flows we have set for each organization"""   
#                         try:
#                             num_covered_flows = len(self.each_testing_wk_each_k_user_pairs[wk_idx][k])
#                         except:
#                             num_covered_flows = 0
#                         if num_covered_flows<self.number_of_flows:
#                             try:
#                                 self.each_testing_wk_each_k_user_pairs[wk_idx][k].append(user_pair)
#                             except:
#                                 self.each_testing_wk_each_k_user_pairs[wk_idx]={}
#                                 self.each_testing_wk_each_k_user_pairs[wk_idx][k]=[user_pair]

#                             """we create an id for this flow and added to the data structure
#                             This is becasue we want to let two organizations have same flows but with different ids"""
#                             try:
#                                 self.each_wk_k_id_pair[wk_idx][k][user_pair_id] = user_pair
#                             except:
#                                 try:
#                                     self.each_wk_k_id_pair[wk_idx][k] ={}
#                                     self.each_wk_k_id_pair[wk_idx][k][user_pair_id] = user_pair
#                                 except:
#                                     self.each_wk_k_id_pair[wk_idx] ={}
#                                     self.each_wk_k_id_pair[wk_idx][k] ={}
#                                     self.each_wk_k_id_pair[wk_idx][k][user_pair_id] = user_pair
#                             try:
#                                 self.each_wk_k_pair_id[wk_idx][k][user_pair] = user_pair_id
#                             except:
#                                 try:
#                                     self.each_wk_k_pair_id[wk_idx][k]={}
#                                     self.each_wk_k_pair_id[wk_idx][k][user_pair] = user_pair_id
#                                 except:
#                                     self.each_wk_k_pair_id[wk_idx]={}
#                                     self.each_wk_k_pair_id[wk_idx][k] ={}
#                                     self.each_wk_k_pair_id[wk_idx][k][user_pair] = user_pair_id
                                    
#                             self.each_testing_user_organization[user_pair_id] = k
#                             self.each_user_pair_id_work_load[user_pair_id] = wk_idx
#                             self.each_user_pair_id_organization[user_pair_id] = k
                        
#     #                         print(" **** adding pair %s # %s for work load %s for user pair %s ****"%(user_pair_id,num_covered_flows,wk_idx,user_pair))
#                             try:
#                                 self.each_testing_wk_each_k_user_pair_ids[wk_idx][k].append(user_pair_id)
#                             except:
#                                 try:
#                                     self.each_testing_wk_each_k_user_pair_ids[wk_idx][k]= [user_pair_id]
#                                 except:
#                                     self.each_testing_wk_each_k_user_pair_ids[wk_idx]={}
#                                     self.each_testing_wk_each_k_user_pair_ids[wk_idx][k]= [user_pair_id]
#                             # we set the weight of each flow
#                             try:
#                                 self.each_testing_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight
#                                 self.each_testing_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight
#                             except:
#                                 try:
#                                     self.each_testing_wk_k_u_weight[wk_idx][k] ={}
#                                     self.each_testing_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight

#                                     self.each_testing_wk_k_u_pair_weight[wk_idx][k] ={}
#                                     self.each_testing_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight
#                                 except:
#                                     self.each_testing_wk_k_u_weight[wk_idx] ={}
#                                     self.each_testing_wk_k_u_weight[wk_idx][k] = {}
#                                     self.each_testing_wk_k_u_weight[wk_idx][k][user_pair_id] = flow_weight

#                                     self.each_testing_wk_k_u_pair_weight[wk_idx] ={}
#                                     self.each_testing_wk_k_u_pair_weight[wk_idx][k] ={}
#                                     self.each_testing_wk_k_u_pair_weight[wk_idx][k][user_pair] = flow_weight

#                             # we set the fidelity threshold of the flow
#                             try:
#                                 self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id] =flow_fidelity_threshold 
#                             except:
#                                 try:
#                                     self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k]={}
#                                     self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id] =flow_fidelity_threshold 
#                                 except:
#                                     self.purification.each_wk_k_u_fidelity_threshold[wk_idx]={}
#                                     self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k]={} 
#                                     self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id] =flow_fidelity_threshold 
                                
#                             """We set the weight of the organization"""
#                             try:
#                                 self.each_testing_wk_k_weight[wk_idx][k] = org_weight
#                             except:
#                                 self.each_testing_wk_k_weight[wk_idx]= {}
#                                 self.each_testing_wk_k_weight[wk_idx][k] = org_weight  
#                             # we set the work load its organization
#                             try:
#                                 if k not in self.each_testing_wk_organizations[wk_idx]:
#                                     try:
#                                         self.each_testing_wk_organizations[wk_idx].append(k)
#                                     except:
#                                         self.each_testing_wk_organizations[wk_idx]=[k]
#                             except:
#                                 self.each_testing_wk_organizations[wk_idx]=[k]
                        
                    
                    
                    
    def set_each_k_user_pair_paths(self,wk_idx):
        """we set self.num_of_paths for each user pair of each organization """
        self.path_counter_id  = 0
        added_paths = []
        self.each_wk_each_k_each_user_pair_id_all_paths={}
        self.each_wk_each_k_each_user_pair_id_paths = {}
        for k,user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair_id in user_pair_ids:
                user_pair = self.each_wk_k_id_pair[wk_idx][k][user_pair_id]
#                 print("for org %s flow id %s pair %s "%(k,user_pair_id,user_pair))
                having_at_least_one_path_flag = False
                one_path_added_flag = False
                for path in self.k_shortest_paths(user_pair[0], user_pair[1], self.num_of_paths,"weight"):
                    node_indx = 0
                    path_edges = []
                    for node_indx in range(len(path)-1):
                        path_edges.append((path[node_indx],path[node_indx+1]))
                        node_indx+=1
#                     flag= False
#                     for p_id,edges in self.set_of_paths.items():
#                         if edges == path_edges:
#                             path_id2 = p_id
#                             flag = True
#                     if flag:
#                         path_id=path_id2
                    path_fidelity = self.purification.get_fidelity(path_edges,self.set_of_virtual_links)
                    #if path_fidelity>0.5:# the condition of 1==1 is for the reasont hat we want to use all paths
                    #if not flag:
#                     pur_scheme  = self.purification.purification_schemes[0]
                
                    if self.distilation_scheme_for_huristic=="random":
                        size_of_distilation_strategies = len(self.purification.set_of_edge_level_Fth)
                        edge_level_Fth = self.purification.set_of_edge_level_Fth[random.randint(0,(size_of_distilation_strategies-1))]
                    elif self.distilation_scheme_for_huristic=="edge-level":
                        edge_level_Fth = 1
                    elif self.distilation_scheme_for_huristic=="end-level":
                        edge_level_Fth=0
                    else:
                        size_of_distilation_strategies = len(self.purification.set_of_edge_level_Fth)
                        edge_level_Fth = self.purification.set_of_edge_level_Fth[random.randint(0,(size_of_distilation_strategies-1))]
                    self.purification.each_path_edge_level_Fth[self.path_counter_id]=edge_level_Fth

                    path_id = self.path_counter_id
                    added_paths.append(path_edges)
                    self.set_each_path_length(self.path_counter_id,path_edges)
                    self.set_of_paths[self.path_counter_id] = path_edges
#                         self.purification.each_path_id_purificaiton_scheme[self.path_counter_id] = pur_scheme
                    #we set the basic fidelity of path here
                    self.purification.each_path_basic_fidelity[path_id]= round(path_fidelity,3)
                    #print("these are the user pair ids in wk %s k %s: %s"%(wk_idx,k,self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k]))
#                         print("2 we set Fth %s for wk %s k %s path %s from flow %s "%(self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id],wk_idx,k,self.path_counter_id,user_pair_id))
                    self.purification.each_path_flow_target_fidelity[self.path_counter_id] = self.purification.each_wk_k_u_fidelity_threshold[wk_idx][k][user_pair_id]
                    try:
                        if len(self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id])<self.num_of_paths:
                            having_at_least_one_path_flag = True
                            one_path_added_flag=True
                            try:
                                self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id].append(path_id)
                            except:
                                try:
                                    self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = [path_id]
                                except:
                                    try:
                                        self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k]={}
                                        self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = [path_id]
                                    except:
                                        self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx]={}
                                        self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k]={}
                                        self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = [path_id]
                    except:
                        having_at_least_one_path_flag = True
                        one_path_added_flag=True
                        try:
                            self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = [path_id]
                        except:
                            try:
                                self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k]={}
                                self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = [path_id]
                            except:
                                self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx]={}
                                self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k]={}
                                self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = [path_id]
                    self.path_counter_id+=1
                if not having_at_least_one_path_flag:
                    try:
                        self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = []
                    except:
                        try:
                            self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = []
                        except:
                            self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx]={}
                            self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k]={}
                            self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id] = []

#                     shortest_disjoint_paths = nx.edge_disjoint_paths(self.g,s=user_pair[0],t=user_pair[1])             
        
    def k_shortest_paths(self, source, target, k, weight):
        return list(
            islice(nx.shortest_simple_paths(self.g, source, target, weight=weight), k)
        )
    
    
    
    def set_distillable_paths(self,wk_idx):
        """here we exclude the paths that after edge-level distilation have e2e fidleity less than 0.5"""
        
        self.each_wk_each_k_each_user_pair_id_paths = {}
        
        for k,user_pair_ids in self.each_wk_each_k_user_pair_ids[wk_idx].items():
            for user_pair_id in user_pair_ids:
                user_pair = self.each_wk_k_id_pair[wk_idx][k][user_pair_id]
                having_at_least_one_path_flag = False
                one_path_added_flag = False
                all_path_ids = self.each_wk_each_k_each_user_pair_id_all_paths[wk_idx][k][user_pair_id]
                for path_id in all_path_ids:
                    if self.purification.check_path_distilability(path_id):
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

    
    def set_paths_in_the_network(self,wk_idx):
        self.reset_pair_paths()
        self.set_each_k_user_pair_paths(wk_idx)
        """we set the required EPR pairs to achieve each fidelity threshold"""
        self.purification.set_required_EPR_pairs_for_distilation(wk_idx,self)
        self.set_distillable_paths(wk_idx)
       
    def reset_pair_paths(self):
        self.path_id = 0
        self.path_counter_id = 0
        self.set_of_paths = {}
        self.purification.each_path_version_numbers= {}
        self.purification.each_path_id_purificaiton_scheme = {}
        self.each_k_u_paths = {}
        self.each_k_u_disjoint_paths = {}
        self.each_k_path_path_id = {}
        self.each_k_u_all_paths = {}
        self.each_k_u_all_disjoint_paths = {}
        self.each_k_path_path_id={}
        
        # these are purification object data sctructures the we reset
        self.purification.each_path_basic_fidelity ={}
        self.purification.all_basic_fidelity_target_thresholds = []
        self.purification.each_n_f_purification_result = {}
        self.purification.each_edge_target_fidelity = {}
        self.purification.function_g_computed  = False
        self.purification.oracle_for_edge_target_fidelity = {}
        self.purification.oracle_for_target_fidelity = {}
        self.purification.each_path_version_numbers= {}
        self.purification.each_path_id_purificaiton_scheme = {}
        self.purification.each_path_flow_target_fidelity = {}
        self.purification.each_path_target_fidelity_for_edge_dislication = {}
        self.purification.unable_to_distill_paths =[]
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
        self.each_wk_k_id_pair ={}
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

    


