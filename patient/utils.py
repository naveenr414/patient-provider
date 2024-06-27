import glob 
import json 
import os 
import resource

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

def restrict_resources():
    """Set the system to only use a fraction of the memory/CPU/GPU available
    
    Arguments: None
    
    Returns: None
    
    Side Effects: Makes sure that a) Only 50% of GPU is used, b) 1 Thread is used, and c) 30 GB of Memory"""

    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))