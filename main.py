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

import time
import os
FLAGS = flags.FLAGS


# In[ ]:


def main(_):
    config = get_config(FLAGS) or FLAGS
    for num_paths in range(int(config.min_num_of_paths),int(config.num_of_paths)+1):
        for edge_capacity_bound in config.edge_capacity_bounds:
            for topology in config.list_of_topologies:
                network = Network(config,topology,edge_capacity_bound,False)
                for fidelity_threshold_up_range in config.fidelity_threshold_ranges:
                    network.fidelity_threshold_range = fidelity_threshold_up_range
                    
                    for edge_fidelity_range in config.edge_fidelity_ranges:
                        print("config.purification_scheme",config.purification_scheme)
                        for purificaion_scheme in config.purification_scheme:
                            if purificaion_scheme =="end_level":
                                network.end_level_purification_flag = True
                            else:
                                network.end_level_purification_flag = False


                            network.set_edge_fidelity(edge_fidelity_range)
                            # we get all the paths for all workloads
                            network.num_of_paths = num_paths

                            for q_value in config.q_values:
                                network.q_value = q_value
                                network.set_nodes_q_value()
                                for number_of_flows in network.number_of_flow_set:
                                    network.number_of_flows = number_of_flows
                                    network.set_flows_of_organizations()
                                    network.set_each_wk_k_fidelity_threshold()
                                    for alpha_value in [0.01]:
                                        network.update_link_rates(alpha_value)
                                        network.get_path_info()
                                        for scheme in config.schemes:
                                            print("for scheme %s flow size %s "%(scheme,number_of_flows))
                                            if scheme in ["EGR","EGRSquare","Hop"]:
                                                network.evaluate_shortest_path_routing(scheme)
                                            elif scheme =="Genetic":
                                                network.evaluate_genetic_algorithm_for_path_selection(config)
                                            elif scheme =="RL":
                                                network.evaluate_rl_for_path_selection(config)
                                            else:
                                                print("not valid scheme (%s): set schemes from EGR, EGRSquare,Hop, Genetic, or RL keywords"%(scheme))


# In[ ]:


# import time
# while(True):
#     print("testing ... :)")
#     time.sleep(10)


# In[ ]:


if __name__ == '__main__':
    app.run(main)

