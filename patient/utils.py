import glob 
import json 
import os 
import resource
import numpy as np
from copy import deepcopy 

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
        load_file = json.load(open(file_name,"r"))

        if 'parameters' in load_file and load_file['parameters'] == data['parameters']:
            try:
                os.remove(file_name)
            except OSError as e:
                print(f"Error deleting {file_name}: {e}")

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
        load_file = json.load(open(file_name,"r"))

        for p in parameters:
            if p not in load_file['parameters'] or load_file['parameters'][p] != parameters[p]:
                break 
        else:
            ret_results.append(load_file)
    return ret_results

def compute_utility(patient_vector,provider_vector,coefficient_vector,context_dim):
    distances = np.abs(patient_vector-provider_vector)
    distances = 1-distances 
    return distances.dot(coefficient_vector)/np.sum(coefficient_vector)

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
        linear_whittle_results = {}
        for key in data_point:
            is_list = False
            if type(data_point[key]) == list and (type(data_point[key][0]) == int or type(data_point[key][0]) == float):
                value = data_point[key][0]
            elif type(data_point[key]) == int or type(data_point[key]) == float:
                value = data_point[key]
            elif type(data_point[key]) == list and type(data_point[key][0]) == list:
                is_list = True 
                value = data_point[key][0]
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
                        data_point[key] = data_point[key]/avg_by_type[data_type]
                        data_point[key] -= 1
                    else:
                        data_point[key][0] /= float(avg_by_type[data_type])

    return aggregate_data(results_copy)