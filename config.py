# This is the configuration file for the experiments. 
class NetworkConfig(object):
    
  """this class would be used in the reinforcement learning implementation"""  
  version = 'RL_v14'
  project_name = 'inter_domain_egr_maximizationv150surfnet14'
  testing_results = "results/juniper_path_selection_evaluationv150surfnet14.csv"
  training_testing_switching_file = "training_testing_switching_file150surfnetv14.txt"
  tf_ckpts = "tf_ckpts150surfnet14"
  method = 'actor_critic'
#   method = 'pure_policy'
  
  model_type = 'Conv'
  scale = 100

  max_step = 50 * scale
  
  initial_learning_rate = 0.0001
  learning_rate_decay_rate = 0.96
  learning_rate_decay_step = 8 * scale
  moving_average_decay = 0.9999
  entropy_weight = 0.2

 
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
  list_of_topologies = ["ModifiedSurfnet.txt"]#"ModifiedSurfnet_wide_range_weights.txt","ModifiedUsCarrier.txt","ModifiedSurfnet.txt"
  multiplexing_value = 1
  work_load_file = 'WK'
  min_edge_capacity = 1# EPR per second
  max_edge_capacity = 400# EPR per second
  min_edge_fidelity = 0.98
  max_edge_fidelity = 0.99
  fidelity_threshold_ranges = [0.9]# for organizations
  edge_fidelity_ranges = [0.9]
  edge_capacity_bounds = [400]
  schemes = ["EGR","Hop","EGRSquare","Genetic","RL"]# Set the schemes that you want to evamluate. If you want to evaluate only genetic algorithm, set only Genetic keyword in the list
  schemes = ["Genetic"]
  purification_scheme =  ["end_level"]
  
  num_of_organizations = 1
  number_of_user_pairs = 1 # Number of user pairs that exist for each organization. 
  min_num_of_paths = 6
  num_of_paths = 6
  cut_off_for_path_searching = 15# We use this cut off when we search for the set of paths that a user pair can use
  q_values = [1]# Different values of q (swap success probability) that we want to evaluate in the experiment
  number_of_work_loads = 1
  number_of_flow_set = [150]# the number of flows that we want to maximize EGR for them
  
    
  
  """parameters of genetic algorithm"""
  population_sizes = [100]
  max_runs_of_genetic_algorithm = 2000
  elit_pop_sizes = [100]
  selection_op_values =[1.0] # values we want to test genetic algorithm
  cross_over_values =[0.5] # values we want to check crossover
  mutation_op_values =[0.1] # values we want to check mutation
  runs_of_genetic_algorithm = 10
  multi_point_mutation_value = 3
  multi_point_crossover_value = 3
  genetic_algorithm_initial_population = "EGR"
  genetic_algorithm_random_initial_population = 20# only this percent of the inial population is random
  ga_elit_pop_update_step = 10
  ga_selection_prob_update_step = 10
  ga_crossover_mutation_update_step = 10
  ga_only_elit_to_generate_population = True
  dynamic_policy_flag = True
    
  flow_path_values_results = "results/flow_path_values_results_250virtual_edges_second_test.csv"
    
  number_of_training_wks = 1
  workloads_to_test = 1
  
  toplogy_wk_scheme_result_file = "results/evaluation_results_wide_range_flow_weights_250virtual_edges_dynamic_policy.csv"
    

def get_config(FLAGS):
  config = Config

  for k, v in FLAGS.__flags.items():
    if hasattr(config, k):
      setattr(config, k, v.value)

  return config
