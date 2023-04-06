# This is the configuration file for the experiments. 
class NetworkConfig(object):
    
  """this class would be used in the reinforcement learning implementation"""  
  version = 'RL_v1'
  project_name = 'inter_domain_egr_maximizationv250surfnet1'
  testing_results = "results/juniper_path_selection_evaluationsurfnet1.csv"
  training_testing_switching_file = "tf_ckpts/training_testing_switching_file_surfnetv6.txt"
  tf_ckpts = "tf_ckpts/tf_ckpts_surfnet6"
  method = 'actor_critic'
#   method = 'pure_policy'
  
  model_type = 'Conv'
  scale = 100

  max_step = 50 * scale
  
  initial_learning_rate = 0.0001
  learning_rate_decay_rate = 0.9
  learning_rate_decay_step_multiplier = 5
  learning_rate_decay_step = learning_rate_decay_step_multiplier * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.2
  rl_batch_size = 3

 
  save_step = 20 
  max_to_keep = 1000

  Conv2D_out = 128
  Dense_out = 128
  
  optimizer = 'RMSprop'
#   optimizer = 'Adam'
    
  logit_clipping = 10       #10 or 0, = 0 means logit clipping is disabled



  test_traffic_file = 'WK2'
  max_moves = 30            #percentage
  # For pure policy
  baseline = 'avg'          #avg, best
#   baseline = 'best'          #avg, best
class Config(NetworkConfig):
  """ this class includes all the experiments setup"""
  #ModifiedSurfnet.txt is for 250 virtual edges
  list_of_topologies = ["ModifiedSurfnet"]#,"ModifiedKdl.txt","ModifiedSurfnetFixed.txt","ModifiedUsCarrier.txt"
  multiplexing_value = 1
  work_load_file = 'WK'
  schemes = ["EGR","Hop","EGRSquare","Genetic","RL"]# Set the schemes that you want to evamluate. If you want to evaluate only genetic algorithm, set only Genetic keyword in the list
  schemes = ["RL"]
    
        
 
  
  set_of_edge_level_Fth= [0.8,0.85,0.88,0.89,0.9,0.92,0.94,0.96,0.97,0.975,0.98,0.988,0.990,0.992,0.994,0.996,0.998]#16 distilation strategies
  set_of_edge_level_Fth= [0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.899,0.90,0.91,0.92,0.93,0.94,0.945,0.95,0.955,0.96,0.965,0.97,0.975,0.98,0.985,0.988,0.989,0.990,0.991,0.992,0.994,0.996,0.998,0.999]#32 distilation strategies
  set_of_edge_level_Fth= [0.86,0.89,0.93,0.96,0.98,0.990,0.994,0.996]#8 distilation strategies
  set_of_edge_level_Fth= [0.8,0.85,0.88,0.89,0.9,0.92,0.94,0.96,0.97,0.975,0.98,0.988,0.990,0.992,0.994,0.996,0.998]
  distilation_scheme_for_huristic = "random"
  num_of_organizations = 3
  min_num_of_paths = 3
  num_of_paths = 3
  candidate_paths_size_for_genetic_alg = 15
  cut_off_for_path_searching = 5# We use this cut off when we search for the set of paths that a user pair can use
  q_values = [1]# Different values of q (swap success probability) that we want to evaluate in the experiment
  alpha_values = [0.2]
  two_qubit_gate_fidelity_set = [0.1,0.2,0.25,0.3,0.8,0.85,0.9,0.92,0.94,0.96,0.98,0.99,1.0]
  two_qubit_gate_fidelity_set = [1.0]
  measurement_fidelity_set =    [0.978,0.96,0.97,0.975,0.978,0.980,0.982,0.984,0.986,0.988,0.990,0.992,0.994,0.996,0.998,0.999]
  measurement_fidelity_set =    [0.978]
  toplogy_wk_setup_feasibility_checking = "results/feasibility_checking_engineering_SURFnet_machine30.csv"
  number_of_work_loads = 100
  number_of_flow_set = [40]# the number of flows that we want to maximize EGR for them
  min_flow_rate = 4
  get_flow_rates_info = False
  flow_path_values_results = "results/QVPN_flow_path_values_evaluation_machine31_final_feasible_initialization_without_rate_constraint.csv"

   
  
  """parameters of genetic algorithm"""
  population_sizes = [100]
  max_runs_of_genetic_algorithm = 2000
  elit_pop_sizes = [100]
  selection_op_values =[1.0] # values we want to test genetic algorithm
  cross_over_values =[1.0] # values we want to check crossover
  mutation_op_values =[1.0] # values we want to check mutation
  runs_of_genetic_algorithm = 1
  multi_point_mutation_value = 30
  multi_point_crossover_value = 30
  genetic_algorithm_initial_population = "Hop"
  genetic_algorithm_initial_population_rl_epoch_number = 2019
  genetic_algorithm_random_initial_population = 50# only this percent of the inial population is random
  ga_elit_pop_update_step = 10
  ga_selection_prob_update_step = 10
  ga_crossover_mutation_update_step = 10
  ga_crossover_mutation_multi_point_update_step = 10
  ga_only_elit_to_generate_population = True
  dynamic_policy_flag = True
  moving_indivituals_to_next_gen = False
  static_crossover_p = 0.5
  static_mutation_p = 0.2
  static_elit_pop_size = 50
  static_multi_point_crossover_value = 30
  static_multi_point_mutation_value =30
        
    
  optimization_problem_with_minimum_rate= False
    
  number_of_training_wks = 2
  workloads_to_test = 2
  workloads_with_feasible_solution = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99]
  save_rl_results_for_initialization = True
#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine4.csv"
#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine1.csv"
#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine15.csv"
  toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_final_machine6_testing_rl.csv"

#   toplogy_wk_rl_for_initialization_ga_result_file = "results/rl_for_initialization_result_file_evaluation_results_evaluating_rl_final_machine8.csv"
  set_of_epoch_for_saving_rl_results_for_ga = [19,39,59,99,519,1019,1519,2019,3019,4019]
  
#   toplogy_wk_scheme_result_file = "results/evaluation_results_dynamic_policy_rl_all_workloads_machine4_final_machine4.csv"
#   toplogy_wk_scheme_result_file = "results/evaluation_results_dynamic_policy_rl_all_workloads_machine4_final_machine1.csv"
#   toplogy_wk_scheme_result_file = "results/evaluation_results_dynamic_policy_rl_all_workloads_machine4_final_machine15.csv"
  toplogy_wk_scheme_result_file = "results/evaluation_results_QVPN_SURFnet_machine6_final_without_rate_constraint_testing_rl.csv"
  

#   toplogy_wk_scheme_result_file = "results/evaluation_results_evaluating_rl_final_machine8.csv"
    

def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
