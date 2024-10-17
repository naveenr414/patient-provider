import glob 
import json 
import os 
import resource
import numpy as np
from copy import deepcopy 
import gurobipy as gp
from gurobipy import GRB

def get_save_path(folder_name,result_name,use_date=False):
    """Create a string, file_name, which is the name of the file to save
    
    Arguments:
        project_name: String, such as 'baseline_bandit'
        seed: Integer, such as 43
        
    Returns: String, such as 'baseline_bandit_43_2023-05-06'"""

    suffix = "{}/{}".format(folder_name,result_name)
    suffix += '.json'
    return suffix

def delete_duplicate_results(folder_name,result_name,data):
    """Delete all results with the same parameters, so it's updated
    
    Arguments:
        folder_name: Name of the results folder to look in
        results_name: What experiment are we running (hyperparameter for e.g.)
        data: Dictionary, with the key parameters
        
    Returns: Nothing
    
    Side Effects: Deletes .json files from folder_name/result_name..."""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))

    for file_name in all_results:
        try:
            f = open(file_name)
            first_few = f.read(1000)
            first_few = first_few.split("}")[0]+"}}"
            load_file = json.loads(first_few)['parameters']
            if load_file == data['parameters']:
                try:
                    os.remove(file_name)
                except OSError as e:
                    print(f"Error deleting {file_name}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

def get_results_matching_parameters(folder_name,result_name,parameters):
    """Get a list of dictionaries, with data, which match some set of parameters
    
    Arguments:
        folder_name: String, which folder the data is located in
        result_name: String, what the suffix is for the dataset
        parameters: Dictionary with key,values representing some known parameters
        
    Returns: List of Dictionaries"""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))
    ret_results = []

    for file_name in all_results:
        try:
            load_file = json.load(open(file_name,"r"))

            for p in parameters:
                if p not in load_file['parameters'] or load_file['parameters'][p] != parameters[p]:
                    break 
            else:
                ret_results.append(load_file)
        except:
            pass
    return ret_results

def restrict_resources():
    """Set the system to only use a fraction of the memory/CPU/GPU available
    
    Arguments: None
    
    Returns: None
    
    Side Effects: Makes sure that a) Only 50% of GPU is used, b) 1 Thread is used, and c) 30 GB of Memory"""

    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))

def aggregate_data(results):
    """Get the average and standard deviation for each key across 
        multiple trials
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""

    ret_dict = {}
    for l in results:
        for k in l:
            if type(l[k]) == int or type(l[k]) == float:
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(l[k])
            elif type(l[k]) == list and (type(l[k][0]) == int or type(l[k][0]) == float):
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(l[k][0])
            elif type(l[k]) == type(np.array([1,2])):
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k] += list(l[k])
            elif type(l[k]) == list:
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k] += list(l[k][0])

    for i in ret_dict:
        ret_dict[i] = (np.mean(ret_dict[i]),np.std(ret_dict[i]))
    
    return ret_dict 

def aggregate_normalize_data(results,baseline=None):
    """Get the average and standard deviation for each key across 
        multiple trials; with each reward/etc. being averaged
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""

    results_copy = deepcopy(results)

    for data_point in results_copy:
        avg_by_type = {}
        for key in data_point:
            is_list = False
            if type(data_point[key]) == list and type(data_point[key][0]) == list:
                is_list = True 
                data_point[key] = np.array(data_point[key][0])
            else:
                continue 
            data_type = key.split("_")[-1]
            if data_type not in avg_by_type and key == "{}_{}".format(baseline,data_type):
                if is_list:
                    avg_by_type[data_type] = np.array(data_point[key])
                else:
                    avg_by_type[data_type] = data_point[key][0]
        if baseline != None:
            for key in data_point:
                data_type = key.split("_")[-1]
                if data_type in avg_by_type:
                    if type(avg_by_type[data_type]) == type(np.array([1,2])):
                        try:
                            data_point[key] = data_point[key][avg_by_type[data_type]!=0] / avg_by_type[data_type][avg_by_type[data_type] !=0]
                        except:
                            continue 
                    else:
                        data_point[key][0] /= float(avg_by_type[data_type])

    return aggregate_data(results_copy)

def solve_linear_program(weights,max_per_provider,lamb=0):
    """Solve a Linear Program which maximizes weights + balance
    
    Arguments:
        weights: Numpy array of size patients x providers
        max_per_provider: Maximum # of patients per provider, >=1
        lamb: Weight placed on the balance objective
    
    Returns: List of tuples, pairs of patient-provider matches"""

    N,P = weights.shape 

    m = gp.Model("bipartite_matching")
    m.setParam('OutputFlag', 0)
    x = m.addVars(N, P, vtype=GRB.BINARY, name="x")

    v = m.addVars(P, name="v")
    beta_bar = m.addVars(1,name="bar")

    m.setObjective(gp.quicksum(weights[i, j] * x[i, j] for i in range(N) for j in range(P)) - lamb/P*gp.quicksum(v[j] for j in range(P)), GRB.MAXIMIZE)
    m.addConstr(beta_bar[0] == 1/P * gp.quicksum(x[i, j] for i in range(N) for j in range(P)))

    for j in range(P):
        m.addConstr(gp.quicksum(x[i, j] for i in range(N)) <= max_per_provider, name=f"match_{j}_limit")

    for i in range(N):
        m.addConstr(gp.quicksum(x[i, j] for j in range(P)) <= 1, name=f"match_{j}")

    m.optimize()

    # Extract the solution
    solution = []
    for i in range(N):
        for j in range(P):
            if x[i, j].X > 0.5:
                solution.append((i, j))
    return solution 

def one_shot_policy(simulator,patient,available_providers,memory,per_epoch_function):
    """Helper function for policies that only need to run once initially
    
    Arguments:
        simulator: Simulator for patient-provider matching
        patient: Particular patient we're finding menu for
        available_providers: 0-1 List of available providers
        memory: Stores which matches, as online policies compute in one-shot fashion
        per_epoch_function: Stores the matches from running the policy once

    Returns: The Menu, from the per epoch function 
    """
    return per_epoch_function[patient.idx], memory 
