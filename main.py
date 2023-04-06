#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function

import numpy as np
import random
import multiprocessing as mp
from absl import app
from absl import flags
import ast
from network import Network
from config import get_config
from solver import Solver
from purification import Purification
import time
import os
FLAGS = flags.FLAGS


# In[ ]:


def main(_):
    config = get_config(FLAGS) or FLAGS
    for num_paths in range(int(config.min_num_of_paths),int(config.num_of_paths)+1):
        for topology in config.list_of_topologies:
            purification = Purification()
            network = Network(config,purification,topology,False)
            
            #network.set_edge_fidelity(edge_fidelity_range)
            # we get all the paths for all workloads
            network.num_of_paths = num_paths
            for gate_fidelity_value in config.two_qubit_gate_fidelity_set:
                network.purification.two_qubit_gate_fidelity = gate_fidelity_value
                for measurement_fidelity in config.measurement_fidelity_set:
                    network.purification.measurement_fidelity = measurement_fidelity
                    network.set_nodes_q_value()
                    for number_of_flows in network.number_of_flow_set:
                        network.number_of_flows = number_of_flows
#                           for alpha_value in [0.0001,0.001,0.05,0.04,0.03,0.02,0.01,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]:
                        for alpha_value in config.alpha_values:
                            network.alpha_value = alpha_value
                            for min_rate in [2]:
                                network.min_flow_rate = min_rate
                                #for target_long_link in [40]:
                                    #for repeater_placement_distance in [10]:
                                        #network.target_long_link=target_long_link
                                        #network.repeater_placement_distance = repeater_placement_distance
                                        #if target_long_link>repeater_placement_distance:
                                            #network.engineer_topology(network.topology_file,target_long_link,repeater_placement_distance)
                                            #import pdb
                                            #pdb.set_trace()
                                network.load_topology()
                                network.update_link_rates(alpha_value)
                                for scheme in config.schemes:
                                    print("for scheme %s flow size %s "%(scheme,number_of_flows))
                                    network.running_path_selection_scheme = scheme
                                    network.set_flows_of_organizations()
                                    print("workloads loaded!")
                                    #network.purification.set_each_wk_k_fidelity_threshold()
                                    network.get_path_info()
                                    print("path info extracted!")
                                    #for edge_elvel_F in [0.8,0.85,0.88,0.89,0.9,0.92,0.94,0.96,0.97,0.975,0.98,0.988,0.990,0.992,0.994,0.996,0.998,0.999,0.9999,0.99999,1.0]:
#                                     network.purification.set_of_edge_level_Fth = []                
#                                     network.purification.set_of_edge_level_Fth = [edge_elvel_F]
#                                     print("network.purification.set_of_edge_level_Fth ",network.purification.set_of_edge_level_Fth)
                                    if scheme in ["EGR","EGRSquare","Hop"]:
                                        network.evaluate_shortest_path_routing(config,scheme)
                                    elif scheme =="Genetic":
                                        network.evaluate_genetic_algorithm_for_path_selection(config)
                                    elif scheme =="RL":
                                        network.evaluate_rl_for_path_selection(config)
                                    else:
                                        print("not valid scheme (%s): set schemes from EGR, EGRSquare,Hop, Genetic, or RL keywords"%(scheme))


# In[ ]:





# In[ ]:


if __name__ == '__main__':
    app.run(main)

