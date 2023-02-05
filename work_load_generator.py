#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


def work_load_generator(topology_file,number_of_work_loads,organization_num,list_of_number_of_flows):
    list_of_number_of_flows.sort()
    for number_of_flows in list_of_number_of_flows:
        g = nx.Graph()
        print('[*] Loading topology...', topology_file)
        try:
            f = open(topology_file+".txt", 'r')
        except:
            f = open(topology_file, 'r')
        header = f.readline()
        nodes = []
        flows = []
        for line in f:
            line = line.strip()
            link = line.split('\t')
            #print(line,link)
            i, s, d,  c,l,real_virtual = link
            if real_virtual =="real":
                if int(s) not in nodes:
                    nodes.append(int(s))
                if int(d) not in nodes:
                    nodes.append(int(d))
        for i in nodes:
            for j in nodes:
                if i!=j and (i,j) not in flows:
                    flows.append((i,j))
        network_max_flows = len(flows)
        selected_candidate_flows = []
        while(len(selected_candidate_flows)<2*number_of_flows):
            candidate_flow = flows[random.randint(0,len(flows)-1)]
            if candidate_flow not in selected_candidate_flows:
                selected_candidate_flows.append(candidate_flow)
        used_flows_for_training = []
        for i in range(number_of_work_loads):
            if i==0 and number_of_flows == min(list_of_number_of_flows):
                lines = ["wk_indx,organization,weight,organization_F,user_pair,weight,flow_F,number_of_flows"+"\n"]
            else:
                lines = []
            
            for organization in range(organization_num):
                selected_flows = []
                each_flow_weight = {}
                each_flow_F_threshold = {}
                org_weight = round(random.uniform(0.3,1),2)
                org_F_threshold = round(random.uniform(0.7,0.92),2)
                while(len(selected_flows)<min(number_of_flows,network_max_flows)):
                    random_flow  = selected_candidate_flows[random.randint(0,len(selected_candidate_flows)-1)]
                    if random_flow not in used_flows_for_training:
                        used_flows_for_training.append(random_flow)
                    if random_flow not in selected_flows:
                        selected_flows.append(random_flow)
                        flow_weight = round(random.uniform(0.3,1.0),2)
                        flow_F = round(random.uniform(0.7,0.92),2)
                        each_flow_weight[random_flow] = flow_weight
                        each_flow_F_threshold[random_flow] = flow_F
                for flow in selected_flows:
                    lines.append(str(i)+","+str(organization)+","+str(org_weight)+","+str(org_F_threshold)+","+str(flow[0])+":"+str(flow[1])+","+str(each_flow_weight[flow])+","+str(each_flow_F_threshold[flow])+","+str(number_of_flows)+"\n")
                file_name = topology_file.split("data/")[1]
                file_name = topology_file.split(".txt")[0]
                with open(str(file_name)+"WK", "a+") as file_object:
                    for line in lines:
                        print("line is ",line)
                        file_object.write(line)
        for  i in range(int(30*number_of_work_loads/100)):  
            if i==0 and number_of_flows == min(list_of_number_of_flows):
                WK2lines = ["wk_indx,organization,weight,organization_F,user_pair,weight,flow_F,number_of_flows"+"\n"]
            else:
                WK2lines = []
            
            for organization in range(organization_num):
                selected_flows = []
                each_flow_weight = {}
                each_flow_F_threshold = {}
                org_weight = round(random.uniform(1,1),2)
                org_F_threshold = round(random.uniform(0.7,0.92),2)
                while(len(selected_flows))<min(number_of_flows,network_max_flows):
                    random_flow  = used_flows_for_training[random.randint(0,len(used_flows_for_training)-1)]
                    if random_flow not in selected_flows:
                        selected_flows.append(random_flow)
                        flow_weight = round(random.uniform(0.3,0.5),2)
                        flow_F = round(random.uniform(0.7,0.92),2)
                        each_flow_weight[random_flow] = flow_weight
                        each_flow_F_threshold[random_flow] = flow_F
                for flow in selected_flows:
                    WK2lines.append(str(i)+","+str(organization)+","+str(org_weight)+","+str(org_F_threshold)+","+str(flow[0])+":"+str(flow[1])+","+str(each_flow_weight[flow])+","+str(each_flow_F_threshold[flow])+","+str(number_of_flows)+"\n")
                file_name = topology_file.split("data/")[1]
                file_name = topology_file.split(".txt")[0]

                with open(str(file_name)+"WK2", "a+") as file_object:
                    for line in WK2lines:
                        print("line is ",line)
                        file_object.write(line)
    


# In[ ]:


work_load_generator("data/ModifiedSurfnet.txt",10,6,[10,20,30,40,50,100,150])


# In[9]:


def parsing_zoo_topologies():
    onlyfiles = [f for f in listdir("data/") if isfile(join("data/", f))]
    not_connected = 0
    connected = 0
    for file in onlyfiles:
        # computing distance between two nodes in graphgml data structure
        #print("this is the file ",file)
        if "graphml" in file and ( "Surfnet" in file and "Modified" not in file):
            G = nx.read_graphml("data/"+file)
            if nx.is_connected(G):
                lines = ["Link_index	Source	Destination	Capacity(EPRps)	Length"+"\n"]
                randomly_selected_nodes = []
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
                for edge in G.edges:
                    #print("for edge ",edge,file)
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
                            print("computed distance is ",distance,coords_1,coords_2)
                            distance = 1
#                         else:
#                             print("one reasonable result")
                    else:
                        distance = round(random.uniform(min(computed_distances),max(computed_distances)),3)
                        #print("a randomly generated among %s %s %s"%(min(computed_distances),max(computed_distances),distance))
                    c = 1
                    etha = 10**(-0.1*0.2*distance)
                    T = (distance*10**(-4))/25# based on simulation setup of data link layer paper
                    if T==0:
                        T=1 * 10**(-6)
                    alpha = 1
                    #print("T is %s for length %s"%(T,distance))
                    distance = round(distance,3)
                    all_distances.append(distance)
                    edge_rate = 5*round((2*c*etha*alpha)/T,3)
                    line = str(edge_indx)+"\t"+str(edge[0])+"\t"+str(edge[1])+"\t"+str(edge_rate)+"\t"+str(distance)+"\t"+"real"+"\n"
                    edge_indx+=1
                    lines.append(line)
                    real_edges.append((edge[0],edge[1]))
                    if edge[0] not in randomly_selected_nodes:
                        randomly_selected_nodes.append(edge[0])
                    if edge[1] not in randomly_selected_nodes:
                        randomly_selected_nodes.append(edge[1])
                    
                covered_edges = []
                for i in range(50):
                    distance = round(random.uniform(min(all_distances),max(all_distances)),2)
                    
                    random_edge = real_edges[random.randint(0,len(real_edges)-1)]
                    node1=random_edge[0]
                    node2=random_edge[1]
                    while(random_edge in covered_edges):
                        random_edge = real_edges[random.randint(0,len(real_edges)-1)]
                        node1=random_edge[0]
                        node2=random_edge[1]
                    new_node1 = node_max_id
                    
                    node_max_id = node_max_id+1
                    covered_edges.append((node1,node2))
                    


                    distance = 0
                    line = str(edge_indx)+"\t"+str(node1)+"\t"+str(new_node1)+"\t"+str(edge_rate)+"\t"+str(distance)+"\t"+"virtual"+"\n"
                    edge_indx+=1
                    lines.append(line)

                    distance = round(random.uniform(min(all_distances),max(all_distances)),2)
                    line = str(edge_indx)+"\t"+str(new_node1)+"\t"+str(node2)+"\t"+str(edge_rate)+"\t"+str(distance)+"\t"+"real"+"\n"
                    edge_indx+=1
                    lines.append(line)

                            
                            
                     # Open the file in append & read mode ('a+')
                print("this is file",file)
                file_name = file.split(".graphml")[0]
                with open("data/ModifiedSurfnet.txt", "a+") as file_object:
                    for line in lines:
                        file_object.write(line)
#                 print("network %s \n has %s nodes \n %s edges \n avg degree %s \n diameter %s \n avg distance %s \n min distance %s \n max distance %s \n unfound nodes %s"%(file,len(G.nodes),len(G.edges), sum(degrees)/len(degrees),diameter, sum(distances)/len(distances),min(distances),max(distances),counter))
#                 print("%s ,  %s ,  %s , , %s , %s , %s , %s , %s "%(file,len(G.nodes),len(G.edges), sum(degrees)/len(degrees),diameter, sum(distances)/len(distances),min(distances),max(distances)))
            else:
                not_connected+=1
            #return 


# In[10]:


#parsing_zoo_topologies()


# In[ ]:




