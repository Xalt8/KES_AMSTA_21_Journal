import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
from pso import PSO, quantity, supply, feasible_vec, split_particles_list, experiment, calculate_profit, poss_val, plot_results

demand = quantity.values


def sib_mix(vec, better_vec):
        ''' Takes 2 vectors and returns the mix of the two
            qb: a number of values that are to be replaced - from
            the total differences between the 2 vectors
            diff_index: the qb number ([:qb]) of index values of the absolute 
            differences between the two vectors sorted in descending order([::-1])
        '''
        mix_vec = vec.copy()
        percent_to_replace = .1
        qb = int(np.ceil(np.where(mix_vec!=better_vec,1,0).sum()*percent_to_replace))
        diff_index = np.argsort(abs(better_vec-mix_vec))[::-1]
        for _ in range(qb):
            for ind in diff_index:
                if poss_val(ind, better_vec[ind], mix_vec ):
                    mix_vec[ind]=better_vec[ind]
                else:
                    break
            
        if feasible_vec(mix_vec):
            return mix_vec
        else:
            print('sib_mix() returned an unfeasible vector')


def random_instantiate_row(demand, supply):
    ''' Helper function for random_jump()
        Takes in demand row values and supply value
        Returns a row that meets the demand and 
        supply constraints.'''
    random_row = np.zeros(demand.shape)
    for j , d in enumerate(demand):
        if d==0:
            continue
        else:
            try:
                r = np.random.randint(0,min(d, supply),1)
                random_row[j]=r
                supply-=r
            except ValueError:
                random_row[j]=0
    
    return random_row


def random_jump(vec):
    ''' Takes a percentage of rows from a matrix and replaces 
        them with random values that meet constraints.
        Parameters:
        Reshapes the vec to demand matrix shape
        percent_to_replace: % rows that will be replaced
        num_rows: gets the numbers of rows to be replaced
        row_indices: randomly selects number of rows (num_rows)
        making sure the same row is not selected more than once
        Replaces the rows by random values & returns vector.'''
    global demand, supply

    vec = vec.reshape(demand.shape)
    percent_to_replace = .4
    num_rows = int(np.ceil(vec.shape[0]*percent_to_replace))
    row_indices = np.random.choice(vec.shape[0], num_rows, replace=False)
    for i in row_indices:
        vec[i] = random_instantiate_row(demand[i], supply[i])
    return vec.flatten()



@dataclass
class SIB(PSO):

    def mix(self):
        for particle in self.particles:
            particle['mixwGB_pos'] = sib_mix(particle['position'], self.gbest_pos)
            particle['mixwGB_val'] = calculate_profit(particle['mixwGB_pos'])
            particle['mixwLB_pos'] = sib_mix(particle['position'], particle['lbest_pos'])
            particle['mixwLB_val'] = calculate_profit(particle['mixwLB_pos'])
    

    def move(self):
        for particle in self.particles:
            if particle['mixwLB_val'] >= particle['mixwGB_val'] >= particle['pbest_val']:
                particle['position'] = particle['mixwLB_pos']
                particle['pbest_val'] = particle['mixwLB_val']
            elif particle['mixwGB_val'] >= particle['mixwLB_val']>= particle['pbest_val']:
                particle['position'] = particle['mixwGB_pos']
                particle['pbest_val'] = particle['mixwGB_val']
            else:
                particle['position'] = random_jump(particle['position'])

def optimise():
    
    start = time.perf_counter()

    iterations = 100
    
    gbest_val_list  = []
    gbest_pos_list  = []
        
    swarm = SIB()
    swarm.initialise()
    swarm.pick_informants_ring_topology()
    
    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()
        swarm.mix()
        swarm.move()

        print(f"Iteration: {i} gbest_val: {round(swarm.gbest_val, 2)}")    

        gbest_val_list.append(round(swarm.gbest_val, 2))
        if i == iterations-1: # Get the value from the last iteration
            gbest_pos_list.append(swarm.gbest_pos)
            if feasible_vec(swarm.gbest_pos):
                print("Constraints met!")

    end = time.perf_counter()
    total_time = end-start

    return gbest_val_list, total_time


if __name__ == '__main__':
    gbest_vals, total_time = optimise()
    plot_results(gbest_vals, total_time)