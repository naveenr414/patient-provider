import json 
import pandas as pd
import numpy as np
from patient.utils import parse_comorbidity_data, get_age_num

cormobidity_list = ['cardio','gastro','neuro','substance','onco'] 
population_rates = {}
specialty_grouping = {
    'cardio': ['CARDIOVASCULAR DISEASE (CARDIOLOGY)','ADVANCED HEART FAILURE AND TRANSPLANT CARDIOLOGY','INTERVENTIONAL CARDIOLOGY','CARDIAC ELECTROPHYSIOLOGY','PERIPHERAL VASCULAR DISEASE','CRITICAL CARE (INTENSIVISTS)'], 
    'gastro': ['NEPHROLOGY','ENDOCRINOLOGY'],
    'neuro': ['CRITICAL CARE (INTENSIVISTS)','GERIATRIC MEDICINE','SLEEP MEDICINE'],
    'substance': ['ADDICTION MEDICINE'], 
    'onco': ['CRITICAL CARE (INTENSIVISTS)','HEMATOLOGY','MEDICAL ONCOLOGY'],
}

for c in cormobidity_list:
    population_rates[c] = parse_comorbidity_data(open("../../data/{}.txt".format(c)).read().split("\n"))

def get_age_costs(ages,age_buckets):
    """Get the cost, a proxy for the workload, based on a patient's age
    
    Arguments:
        ages: The buckets for ages we're working with
        age_buckets: The relative proportion of individuals in each age bucket
    
    Returns: Dictionary, the cost per capita, normalized, for each age"""

    age_info = pd.read_csv(open("../../data/age_cost.csv"))

    age_total_costs = list(age_info['cost'])
    age_cutoffs = list(zip(list(age_info['lower']),list(age_info['upper'])))
    corresponding_bucket = []
    for i in range(len(age_cutoffs)):
        tot = 0
        for j in range(len(ages)):
            if age_cutoffs[i][0] <= ages[j]<=age_cutoffs[i][1]:
                tot += age_buckets[j] 
        corresponding_bucket.append(tot)
    cost_per = np.array(age_total_costs)/np.array(corresponding_bucket)
    cost_per /= np.max(cost_per)

    cost_per_age = {}
    for i in range(len(ages)):
        for j in range(len(cost_per)):
            if age_cutoffs[j][0] <= ages[i] <= age_cutoffs[j][1]:
                cost_per_age[ages[i]] = cost_per[j]

    return cost_per_age

def generate_semi_synthetic_theta_workload(num_patients,num_providers,comorbidities=False):
    """Generate a semi synthetic dataset based on medicare
    Sample providers from the medicare dataset, and place
    patients randomly in different locations in CT
    
    Arguments:
        num_patients: int, the number of patients in \theta
        num_providers: int, the number of providers in \theta
    
    Returns: Two things: 
        1) \theta, Numpy array of size num_patients x num_providers
        2) workload, patient-by-patient workloads"""
    
    medicare_data = pd.read_csv("../../data/medicare_data.csv")
    zipcode_data = pd.read_csv("../../data/connecticut_zipcode.csv")
    age_distro = pd.read_csv("../../data/ct_age.csv")
    zip_code_distances = json.load(open("../../data/ct_zipcode_distance.json"))
    theta = np.zeros((num_patients,num_providers))

    ct_data = medicare_data[medicare_data['State'] == 'CT']
    ct_data = ct_data[ct_data['sec_spec_all'].str.contains('INTERNAL MEDICINE|GENERAL PRACTICE|FAMILY', case=False, na=False)]
    
    if comorbidities:
        for key, specialties in specialty_grouping.items():
            ct_data[key] = ct_data['sec_spec_all'].apply(lambda x: any(spec in x for spec in specialties))

    downsample_ct_data =  ct_data.sample(frac=num_providers/len(ct_data),replace=True)
    
    zipcodes = list(zipcode_data['Zipcode'])
    zipcodes = ["0"*(5-len(str(i)))+str(i) for i in zipcodes]
    population = [int(i.replace(",","")) for i in list(zipcode_data['Population'])]
    zipcode_probabilities = np.array(population)/sum(population)
    age_buckets = []
    ages = []
    for i in range(0,100,5):
        ages.append(i+2.5)
        age_buckets.append(sum(age_distro[(age_distro['Year'] == 2022) & (age_distro['ID Age'] == i)]['Total Population']))
    age_buckets
    age_buckets = [age_buckets[i] for i in range(len(age_buckets)) if ages[i]>=18]
    ages = [i for i in ages if i>=18]
    age_buckets = np.array(age_buckets)/sum(age_buckets)

    random_patients = []

    for i in range(num_patients):
        age = np.random.choice(ages,p=age_buckets)
        location = np.random.choice(zipcodes,p=zipcode_probabilities)

        if comorbidities:
            patient_dict = {'age': age, 'location': location}
            age_group = get_age_num(age)

            for comorbidity in population_rates:
                patient_dict[comorbidity] = int(np.random.random()<population_rates[comorbidity][age_group])
            random_patients.append(patient_dict)
        else:
            random_patients.append({'age': age, 'location': location})

    for i in range(num_patients):
        max_distance = np.random.poisson(20.2)
        beta = max_distance/2
        our_zip = random_patients[i]
        noise = np.random.normal(0,0.1)
        for j in range(len(downsample_ct_data)):
            other_zip = str(downsample_ct_data.iloc[j,:]['ZIP Code'])
            other_zip = (('0'*(9-len(other_zip)) + other_zip)[:5])
            distance = zip_code_distances[str((our_zip['location'], other_zip))]
            distance += 0.01
            if distance <= max_distance:
                distance_utility = min(max(beta/distance + noise,0),1)
                if comorbidities:
                    our_cormobidities = np.array([int(our_zip[d]) for d in cormobidity_list])
                    provider_cormobidities = np.array([int(downsample_ct_data.iloc[j,:][d]) for d in cormobidity_list])
                    cormobidity_score = our_cormobidities.dot(provider_cormobidities)

                    if cormobidity_score > 0:
                        utility = 0.75 + distance_utility*0.25
                    else:
                        utility = 0.5 + distance_utility*0.25
                    theta[i,j] = utility
                else:
                    theta[i,j] = distance_utility
    age_costs = get_age_costs(ages,age_buckets)
    patient_costs = np.array([max(min(age_costs[i['age']]+np.random.normal(0,0.1),1),0) for i in random_patients])
    return theta, patient_costs, random_patients, downsample_ct_data.to_dict('records')