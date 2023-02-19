#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


class Purification:
    def __init__(self):
        self.each_path_basic_fidelity ={}
        self.each_wk_k_fidelity_threshold = {}
        self.oracle_for_target_fidelity = {}
        self.global_each_basic_fidelity_target_fidelity_required_EPRs = {}
        self.all_basic_fidelity_target_thresholds = []
        self.each_n_f_purification_result = {}
        self.each_edge_target_fidelity = {}
        self.each_wk_k_u_fidelity_threshold = {}
        self.each_edge_fidelity = {}

        # we want to compute function g only once
        self.function_g_computed  = False
    
    
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
        if self.function_g_computed:
            #print("not computing the function g!")
            pass
        else:
            self.function_g_computed = True
            targets = []
            for k,u_target_fidelity in self.each_wk_k_u_fidelity_threshold[wk_idx].items():
                for u,target_fidelity in u_target_fidelity.items():
                    if target_fidelity not in targets:
                        targets.append(target_fidelity)
            targets.append(0.6)
            targets.append(0.55)
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
        longest_p_lenght = 0
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
    
    def get_each_wk_k_u_threshold(self,wk_idx,k,u):
        return self.each_wk_k_u_fidelity_threshold[wk_idx][k][u]
    
    def get_fidelity(self,path_edges,virtual_links):
        if path_edges:
            F_product = (4*self.each_edge_fidelity[path_edges[0]]-1)/3 
            for edge in path_edges[1:]:
                if edge not in virtual_links and (edge[1],edge[0]) not in virtual_links:
                    F_product  = F_product*(4*self.each_edge_fidelity[edge]-1)/3

        else:
            print("Error")
            return 1.0
        N = len(path_edges)+1
        p1 = 1
        p2 = 1
        F_final = 1/4*(1+3*(p1*p2)**(N-1)*(F_product))
        return round(F_final,3)
    

