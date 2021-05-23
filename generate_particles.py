import pandas as pd
import numpy as np
import random
import joblib

## Imports
xls = pd.ExcelFile('Data.xlsx')
products = pd.read_excel(xls, 'products', index_col=0, nrows=78, usecols='A:G')
customers = pd.read_excel(xls, 'customers', index_col=0, usecols='A:C')
quantity = pd.read_excel(xls, 'demand_boxes', index_col=0, nrows=78, usecols='A:CE').fillna(0).astype(np.int64)
prices = pd.read_excel(xls, 'price_per_box', index_col=0, nrows=78,usecols='A:CE')
transport = pd.read_excel(xls, 'transport_lookup', index_col=0)
transport = transport.iloc[2:]
transport.drop(transport.columns[[0,1]], axis=1,inplace=True)
supply = products['Qty(eggs)']/products['eggs/pack']
cost_eggs = np.sum(products['Qty(eggs)'].values * products['Cost/egg'].values)


## Functions ##########

def random_instantiates_vector(demand, supply):
    ''' Takes demand and supply constraints and returns a vector
        with random values that meet demand and supply constraints.'''
    mat = np.zeros(demand.shape)
    for d_row, s, mat_row in zip(demand, supply, mat):
        for j , d in enumerate(d_row):
            if d==0:
                continue
            else:
                try:
                    r = np.random.randint(0,min(d, s),1)
                    mat_row[j]=r
                    s-=r
                except ValueError:
                    mat_row[j]=0

    return mat.flatten()


def feasible_vec(vec, demand, supply):
    ''' Checks if the vector meets demand and supply
        constraints -> bool'''
    mat = vec.reshape(demand.shape) 
    demand_check = np.all((mat <= demand) & (mat >=0))
    supply_check = np.all((mat.sum(axis=1)<=supply) & (mat.sum(axis=1)>=0))
    return demand_check and supply_check 


def already_there(list_of_particles, new_particle):
    ''' Checks to see if a new_particle is present in
        list_of_particles -> bool'''
    return np.any([np.array_equal(new_particle, particle) for particle in list_of_particles])


def generate_particle_list(num_of_particles, demand, supply):
    particle_list = []
    while len(particle_list) < num_of_particles:
        vec = random_instantiates_vector(demand, supply)
        if not already_there(particle_list, vec):
            particle_list.append(vec)
        else:
            print("Already there!")
    return particle_list


particle_list = generate_particle_list(2000, quantity.values, supply.values)

joblib.dump(particle_list, 'particle_list')

print(f'Created: {len(particle_list)} particles')