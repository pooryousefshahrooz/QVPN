#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import os
import sys
from docplex.mp.progress import *
from docplex.mp.progress import SolutionRecorder
import docplex.mp.model as cpx
import networkx as nx
import time
from config import get_config
from absl import flags
FLAGS = flags.FLAGS


# In[1]:


class Solver:
    def __init__(self):
        pass
    def CPLEX_maximizing_EGR(self,wk_idx,network):
        #print("scheme ",network.)
#         for k in network.each_wk_organizations[wk_idx]:
#             for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]:                
#                 print("we are k %s u %s paths %s # paths %s allowed %s"%(k,u,network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u],len(network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]),network.num_of_paths))
#                 for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]:
#                     print("wk %s k %s w %s user %s w %s path %s"%(wk_idx,k,network.each_wk_k_weight[wk_idx][k],u,network.each_wk_k_u_weight[wk_idx][k][u],p))
#                     print("edges of the path",network.set_of_paths[p])
#         print("done!")
    
        #print("network.max_edge_capacity",network.max_edge_capacity,type(network.max_edge_capacity))
        opt_model = cpx.Model(name="inter_organization_EGR")
        x_vars  = {(k,p): opt_model.continuous_var(lb=0, ub= network.max_edge_capacity,
                                  name="w_{0}_{1}".format(k,p))  for k in network.each_wk_organizations[wk_idx]
                   for u in network.each_wk_each_k_user_pair_ids[wk_idx][k] for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]}

        
    #     for k in network.K:
    #         for u in network.each_k_user_pairs[k]:
    #             for p in network.each_k_u_paths[k][u]:
    #                 print("organization %s user %s #paths %s cost %s p %s path %s"%(k,u,network.num_of_paths,network.each_link_cost_metric,p,network.set_of_paths[p]))

    #     time.sleep(9)
       
        #Edge constraint
        for edge in network.set_E:
            if network.end_level_purification_flag:
                opt_model.add_constraint(
                    opt_model.sum(x_vars[k,p]*network.each_wk_k_u_weight[wk_idx][k][u] 
                    * network.get_required_purification_EPR_pairs(p,network.get_each_wk_k_threshold(wk_idx,k))
                    for k in network.each_wk_organizations[wk_idx] for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]
                    for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]
                    if network.check_path_include_edge(edge,p))
                     <= network.each_edge_capacity[edge], ctname="edge_capacity_{0}".format(edge))
                
            else:
                opt_model.add_constraint(
                    opt_model.sum(x_vars[k,p]*network.each_wk_k_u_weight[wk_idx][k][u] *
                    network.get_required_edge_level_purification_EPR_pairs(edge,p,network.each_wk_k_fidelity_threshold[k],wk_idx)
                    for k in network.each_wk_organizations[wk_idx] for u in network.each_wk_each_k_user_pair_ids[wk_idx][k]
                    for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]
                    if network.check_path_include_edge(edge,p))
                     <= network.each_edge_capacity[edge], ctname="edge_capacity_{0}".format(edge))
#                 print("from edge %s to F %s divide capacity by this %s"%(edge,network.each_wk_k_fidelity_threshold[0],
#                           network.get_required_edge_level_purification_EPR_pairs
#                           (edge,0,network.each_wk_k_fidelity_threshold[0],0)))

        objective = opt_model.sum(x_vars[k,p]*network.each_wk_k_weight[wk_idx][k] * network.each_wk_k_u_weight[wk_idx][k][u]*network.q_value**(network.get_path_length(p)-1)
                              for k in network.each_wk_organizations[wk_idx]
                              for u in network.each_wk_each_k_user_pair_ids[wk_idx][k] 
                              for p in network.each_wk_each_k_each_user_pair_id_paths[wk_idx][k][u]
                              )


        # for maximization
        opt_model.maximize(objective)

    #     opt_model.solve()
        #opt_model.print_information()
        #try:
        opt_model.solve()


        #print('docplex.mp.solution',opt_model.solution)
        objective_value = -1
        try:
            if opt_model.solution:
                objective_value =opt_model.solution.get_objective_value()
        except ValueError:
            print(ValueError)

        opt_model.clear()

        return objective_value
    
    def CPLEX_swap_scheduling(self,wk_idx,network):
        
        opt_model = cpx.Model(name="swap_scheduling")
        eflow_vars  = {(i,j,k,b): opt_model.continuous_var(lb=0, ub= network.max_edge_capacity,
                        name="eflow_{0}_{1}_{2}_{3}".format(i,j,k,b))
                       for i in network.nodes for j in network.nodes for k in network.nodes for b in network.nodes}

        u_vars  = {(i,j): opt_model.continuous_var(lb=0, ub= 1,
                                  name="u_{0}_{1}".format(i,j)) for i in network.nodes for j in network.nodes}
        
        # We set the capacity of pair of nodes to zero if the pair is not an edge in the network
        for i in network.nodes:
            for j in network.nodes:
                if (i,j) not in network.set_E:
                    network.each_edge_capacity[(i,j)] = 0
        
        # we check incoming and out going flow to each enode
        """we divide the edge capacity by the g(.) for the higherst threshold to count edge level purification.
        
        It is not clear how would we consider end level purification."""
        #print("this is the end level purification flag ",network.end_level_purification_flag)
        if not network.end_level_purification_flag:
            for i in network.nodes:
                for j in network.nodes:
                    #print("we are adding incoming outgoing flow constraint")
                    if i!=j:
                        opt_model.add_constraint(opt_model.sum(network.each_node_q_value[k] *
                        (eflow_vars[i,k,i,j]+eflow_vars[k,j,i,j])/2 for k in network.nodes if k not in [i,j])
                        +(network.each_edge_capacity[(i,j)] 
                         * network.check_edge_exit((i,j))) 
                        - opt_model.sum(
                        (eflow_vars[i,j,i,k]+eflow_vars[i,j,k,j]) for k in network.nodes if k not in [i,j])
                        >=0)
                        #With average round of purification required on paths
#                         opt_model.add_constraint(opt_model.sum(network.each_node_q_value[k] *
#                         (eflow_vars[i,k,i,j]+eflow_vars[k,j,i,j])/2 for k in network.nodes if k not in [i,j])
#                         +(network.each_edge_capacity[(i,j)] 
#                          * network.check_edge_exit((i,j)))
#                         /network.get_required_edge_level_purification_EPR_pairs_all_paths((i,j),i,j,wk_idx) 
#                         - opt_model.sum(
#                         (eflow_vars[i,j,i,k]+eflow_vars[i,j,k,j]) for k in network.nodes if k not in [i,j])
#                         >=0)
        for i in network.nodes:
            for j in network.nodes:
                for k in network.nodes:
                    opt_model.add_constraint(eflow_vars[i,k,i,j]==eflow_vars[k,j,i,j])
                    opt_model.add_constraint(eflow_vars[i,k,i,j]>=0)
        for k in network.each_wk_organizations[wk_idx]:
            for u in network.each_wk_each_k_user_pairs[wk_idx][k]:
                for i in network.nodes:
                    opt_model.add_constraint(eflow_vars[u[0],u[1],u[0],i]==eflow_vars[u[0],u[1],u[1],i])
                    opt_model.add_constraint(eflow_vars[u[0],u[1],u[0],i]==0)



        objective = opt_model.sum((network.each_edge_capacity[(u[0],u[1])] * network.check_edge_exit((u[0],u[1]))
                        + opt_model.sum(network.each_node_q_value[k] * 
                        (eflow_vars[u[0],k,u[0],u[1]]+eflow_vars[k,u[1],u[0],u[1]])/2 for k in network.nodes 
                            if k not in [u[0],u[1]]) )
                            *network.each_wk_k_weight[wk_idx][k] * network.each_wk_k_u_pair_weight[wk_idx][k][u]
                            for k in network.each_wk_organizations[wk_idx]
                            for u in network.each_wk_each_k_user_pairs[wk_idx][k] 
                              )


        # for maximization
        opt_model.maximize(objective)
    #     opt_model.solve()
        #opt_model.print_information()
        #try:
        opt_model.solve()
        #print('docplex.mp.solution',opt_model.solution)

        objective_value = -1
        try:
            if opt_model.solution:
                objective_value =opt_model.solution.get_objective_value()
        except ValueError:
            print(ValueError)

        #print("EGR is ",objective_value)
        return objective_value


# In[ ]:




