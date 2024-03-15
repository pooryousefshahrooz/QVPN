# This is the configuration file for the experiments. 
class NetworkConfig(object):
    
  """this class would be used in the reinforcement learning implementation"""  
  version = 'RL_v26'
  project_name = 'machine1_B_20_128X128_LRDR096_Ent01v26'
  testing_results = "results/juniper_path_selection_evaluationsurfnet26.csv"
  training_testing_switching_file = "tf_ckpts/training_testing_switching_file_surfnetv26.txt"
  tf_ckpts = "tf_ckpts/tf_ckpts_surfnetv26"
  method = 'actor_critic'
#   method = 'pure_policy'
  
  model_type = 'Conv'
  scale = 100

  max_step = 80 * scale
  
  initial_learning_rate = 0.0001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step_multiplier = 5
  learning_rate_decay_step = learning_rate_decay_step_multiplier * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.1
  rl_batch_size = 20
  
 
  save_step = 20 
  max_to_keep = 1000

  Conv2D_out = 128
  Dense_out = 128
#   Conv2D_out = 255
#   Dense_out = 255
#   Conv2D_out = 224
#   Dense_out = 224
  
  optimizer = 'RMSprop'
#   optimizer = 'Adam'
    
  logit_clipping = 10       #10 or 0, = 0 means logit clipping is disabled



  test_traffic_file = 'WK2'
  max_moves = 30            #percentage
  # For pure policy
  baseline = 'avg'          #avg, best
#   baseline = 'best'          #avg, best
  training = False
class Config(NetworkConfig):
  """ this class includes all the experiments setup"""

   #The main topology that we have engineered and added paralel link and done all the experiments "ModifiedSurfnet.txt"
   # the main results with max 1,000 and min 10 is with ModifiedSurfnet_Eng_7hops_flowsv2

  list_of_topologies = ["ModifiedSurfnet"]#"ModifiedSurfnet_Eng_7hops_flowsv2","ModifiedSurfnet_Eng_7hops_flowsv2_wider_flows","ModifiedSurfnet_Eng_7hops_flowsv2_wider_flowsFths","ModifiedSurfnetv2.txt"
  multiplexing_value = 1
  work_load_file = 'WK'
  schemes = ["EGR","Hop","EGRSquare","Genetic","RL"]# Set the schemes that you want to evamluate. If you want to evaluate only genetic algorithm, set only Genetic keyword in the list
  schemes = ["Genetic"]
    
        
 
  
  set_of_edge_level_Fth= [0.8,0.85,0.88,0.89,0.9,0.92,0.94,0.96,0.97,0.975,0.98,0.988,0.990,0.992,0.994,0.996,0.998]#16 distilation strategies
  set_of_edge_level_Fth= [0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.899,0.90,0.91,0.92,0.93,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.988,0.989,0.990,0.991,0.992,0.994,0.996,0.998,0.999]#32 distilation strategies
  set_of_edge_level_Fth= [0.88,0.9,0.93,0.96,0.98,0.990,0.994,0.996]#8 distilation strategies
#   set_of_edge_level_Fth= [0.86,0.89,0.90,0.92,0.94,0.96,0.98,0.990,0.994,0.996]#10 distilation strategies
  set_of_edge_level_Fth= [0.8,0.85,0.90,0.992]#4 distilation strategies
#   set_of_edge_level_Fth= [0.8,0.9,0.94,0.98,0.992,0.998]#6 distilation strategies
#   set_of_edge_level_Fth= [0.89,0.99]#2 distilation strategies
#   set_of_edge_level_Fth= [0.8,0.85,0.88,0.89,0.9,0.92,0.94,0.96,0.97,0.98,0.988,0.990,0.992,0.994,0.996,0.998]
#   set_of_edge_level_Fth= [0.992]
  distilation_scheme_for_huristic = "random"
  num_of_organizations = 3
  min_num_of_paths = 3
  num_of_paths = 3
  candidate_paths_size_for_genetic_alg = 15
  candidate_path_sizes_for_genetic_alg = [3,5,8,10,15]
  each_cut_of_upper_value = {3:8,5:10,8:15,10:15,15:15}
  cut_off_for_path_searching = 5# We use this cut off when we search for the set of paths that a user pair can use
  toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_final.csv"
  toplogy_wk_paths_for_each_candidate_size = "results/qVPN_paths_for_each_candidate_size_final.csv"

#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_max100final.csv"
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25.csv"
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25v2.csv"
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25v3.csv"
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25v4.csv"
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25v6.csv"# for 7 hops max [10,500]
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25v7.csv"# for 7 hop max [10,1000]
# #   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_april25v8.csv"# for 7 hop max [10,10000]
#   toplogy_wk_paths_for_each_candidate_size = "results/toplogy_wk_paths_for_each_candidate_size_different_min_no_max.csv"# for 7 hop max [10,5000]
  saving_used_paths_flag = False
  q_values = [1]# Different values of q (swap success probability) that we want to evaluate in the experiment
  alpha_values = [0.2]
  two_qubit_gate_fidelity_set = [0.1,0.2,0.25,0.3,0.8,0.85,0.9,0.92,0.94,0.96,0.98,0.99,1.0]
  two_qubit_gate_fidelity_set = [1.0]
  measurement_fidelity_set =    [0.978,0.96,0.97,0.975,0.978,0.980,0.982,0.984,0.986]
  measurement_fidelity_set =    [0.978]
  toplogy_wk_setup_feasibility_checking = "results/feasibility_checking_engineering_SURFnet_machine30.csv"
  number_of_work_loads = 100
  number_of_flow_set = [50]# the number of flows that we want to maximize EGR for them
  min_flow_rate = 10
  get_flow_rates_info = False
  flow_path_values_results = "results/QVPN_flow_values_with_min_max_rate_final.csv"# for with max and min rate constraint
  flow_path_values_results = "results/QVPN_flow_values_without_max_with_min_rate_final.csv"# for without max rate with min rate
  flow_path_values_results = "results/QVPN_flow_values_without_max_and_min_rate_final.csv"# for without neither max nor min
    
    
  flow_path_values_results = "results/QVPN_flow_values_experimenting_fairness_with_min_max_rate.csv"# for fairness experiment
  flow_path_values_results = "results/QVPN_flow_values_experimenting_fairness_with_min_without_max_rate.csv"# for fairness experiment
  flow_path_values_results = "results/QVPN_flow_values_experimenting_fairness_without_min_max_rate.csv"# for fairness experiment
  flow_path_values_results = "results/QVPN_flow_values_experimenting_fairness_wit_min_max_100rate.csv"# for fairness experiment
    
  flow_path_values_results = "results/QVPN_flow_values_experimenting_fairness_with_different_min_with_40k_max_rate.csv"# for fairness experiment
  flow_path_values_results = "results/QVPN_flow_values_experimenting_fairness_with_min_rate_10_with_max_1krate_shortest_path.csv"# for fairness experiment

  flow_path_values_results = "results/qVPN_flow_values_with_min_rate_10_with_max_random1k_mach31.csv"# for fairness experiment
  flow_path_values_results = "results/qVPN_flow_values_with_min_rate_10_with_max_random1k_mach33.csv"# for fairness experiment
#   flow_path_values_results = "results/qVPN_flow_values_with_min_rate_10_with_max_random1k_mach7.csv"# for fairness experiment
#   flow_path_values_results = "results/qVPN_flow_values_without_min_max_random1k_mach26.csv"# for fairness experiment

#   flow_path_values_results = "results/QVPN_flow_values_machine15_without_rate_constraint.csv"
#   flow_path_values_results = "results/QVPN_flow_values_machine35_final_result_with_rate_constraint.csv"

   
  
  """parameters of genetic algorithm"""
  population_sizes = [100]
  max_runs_of_genetic_algorithm = 1000
  elit_pop_sizes = [100]
  selection_op_values =[1.0] # values we want to test genetic algorithm
  cross_over_values =[1.0] # values we want to check crossover
  mutation_op_values =[1.0] # values we want to check mutation
  runs_of_genetic_algorithm = 3
  multi_point_mutation_value = 50
  multi_point_crossover_value = 50
  genetic_algorithm_initial_population = "Hop"
  genetic_algorithm_initial_population_rl_epoch_number = 2019
  genetic_algorithm_random_initial_population = 10# only this percent of the inial population is random
  ga_elit_pop_update_step = 10
  ga_selection_prob_update_step = 10
  ga_crossover_mutation_update_step = 10
  ga_crossover_mutation_multi_point_update_step = 10
  ga_only_elit_to_generate_population = True
  dynamic_policy_flag = True
  moving_indivituals_to_next_gen = True
  static_crossover_p = 0.5
  static_mutation_p = 0.2
  static_elit_pop_size = 50
  static_multi_point_crossover_value = 50
  static_multi_point_mutation_value =50
        
  each_flow_minimum_rate_value_file  ="results/each_flow_min_rate_from_min_10_rate_max1k_ratev5.csv"
  previous_run_path_info_file = "results/QVPN_flow_values_experimenting_fairness_with_min_rate_10_with_max_ratev5.csv"
  set_paths_from_file = False
  get_flows_minimum_rate_flag = False
  optimization_problem_with_minimum_rate= False
  min_flow_rates = [10]
  optimization_problem_with_maximum_rate= False
  up_max_rate_value = 1000
  updating_max_rate_flag = False
  max_flow_rate = 10
    
  number_of_training_wks = 5
  workloads_to_test = 5
  wkidx_min_value = 0
  wkidx_max_value = 5
  workloads_with_feasible_solution = [0,1,2,3,4]
  save_rl_results_for_initialization = False
#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine4.csv"
#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine1.csv"
#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine15.csv"
  toplogy_wk_rl_for_initialization_ga_result_file = "results/gradient_scheme_for_ga_initialization_results_machine1.csv"

#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_evaluation_results_evaluating_rl_final_machine8.csv"
  set_of_epoch_for_saving_rl_results_for_ga = [19,39,59,99,519,1019,1519,2019,3019,4019,4999]
  
#   toplogy_wk_scheme_result_file = "results/evaluation_results_dynamic_policy_rl_all_workloads_machine4_final_machine4.csv"
#   toplogy_wk_scheme_result_file = "results/evaluation_results_dynamic_policy_rl_all_workloads_machine4_final_machine1.csv"
#   toplogy_wk_scheme_result_file = "results/evaluation_results_dynamic_policy_rl_all_workloads_machine4_final_machine15.csv"
  
    
    
    
  toplogy_wk_scheme_result_file = "results/QVPN_machine22_with_min_max_rate_results_final.csv"
  toplogy_wk_scheme_result_file = "results/QVPN_machine27_without_max_rate_with_min_rate_constraint_results.csv"
  toplogy_wk_scheme_result_file = "results/QVPN_machine27_without_max_and_min_rate_constraint_results.csv"

  toplogy_wk_scheme_result_file = "results/QVPN_machine27_experimenting_fairness_with_min_max_rate_results.csv"
  toplogy_wk_scheme_result_file = "results/QVPN_machine27_experimenting_fairness_with_min_without_max_rate_results.csv"
  toplogy_wk_scheme_result_file = "results/QVPN_machine19_experimenting_fairness_with_min_max_100rate_results.csv"
  toplogy_wk_scheme_result_file = "results/QVPN_machine27_experimenting_fairness_with_different_min_with_2kmax_rate_results_shortest_path.csv"

#   toplogy_wk_scheme_result_file = "results/QVPN_machine34_without_rate_constraint_results_shortest_path_schemes.csv"

#   toplogy_wk_scheme_result_file = "results/QVPN_machine31_without_rate_constraint_one_org_results.csv"
#   toplogy_wk_scheme_result_file = "results/QVPN_machine1_without_rate_constraint_one_org_random_ga_ini_results.csv"
      

    
    
  toplogy_wk_scheme_result_file = "results/qVPN_final_results_machine32.csv"
  toplogy_wk_scheme_result_file = "results/qVPN_final_results_wider_flowsmachine37.csv"
  toplogy_wk_scheme_result_file = "results/qVPN_final_results_machine33_for_fairness.csv"
  toplogy_wk_scheme_result_file = "results/qVPN_final_results_machine33_for_distillation_old_topology.csv"
#   toplogy_wk_scheme_result_file = "results/qVPN_final_results_machine7_for_fairness.csv"
    

def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
