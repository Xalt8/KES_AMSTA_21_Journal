
import numpy as np
import pandas as pd
from dataclasses import dataclass
import random
import time
import matplotlib.pyplot as plt
import cProfile, pstats, io # Needed for profile
from pso import PSO, random_back, quantity, supply, feasible_vec, plot_results, split_particles_list, make_result_matrix


demand = quantity.values

@dataclass
class CPSO(PSO):
    
    def set_constricted_velocity(self):
        for particle in self.particles:
            c1 = 2.05
            c2 = 2.05
            ep = c1+c2
            X = 2/(abs(2-ep-np.sqrt((ep**2)-4*ep)))
            dims = particle['position'].shape
            cognitive = (c1 * np.random.uniform(0, 1, dims)*(particle['pbest_pos'] - particle['position']))
            informers = (c2 * np.random.uniform(0, 1, dims)*(particle['lbest_pos'] - particle['position']))
            new_velocity = X*(particle['velocity'] + cognitive + informers)
            particle['velocity'] = new_velocity

    
    def move_random_back(self):
        for particle in self.particles:
            new_pos = random_back(particle['position'], particle['velocity'])
            particle['position'] = np.floor(new_pos)

    
def optimise(demand, supply):

    start = time.perf_counter()

    iterations = 10
    
    gbest_val_list  = []
    gbest_pos_list  = []
    
    swarm = CPSO()
    swarm.initialise(demand, supply)
    swarm.pick_informants_ring_topology()
    
    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()  
        swarm.set_constricted_velocity()
        swarm.move_random_back()

        print(f"Iteration: {i} gbest_val: {round(swarm.gbest_val, 2)}")    

        gbest_val_list.append(round(swarm.gbest_val, 2))
        if i == iterations-1: # Get the value from the last iteration
            gbest_pos_list.append(swarm.gbest_pos)
            if feasible_vec(swarm.gbest_pos, demand, supply):
                print("Constraints met!")
    
    end = time.perf_counter()
    total_time = round((end-start),2)

    return gbest_val_list, total_time


def experiment(split_particles_list, experiment_name):
    
    results = [optimise(demand, supply) for i in range(len(split_particles_list))]

    experiment_results = make_result_matrix(results)
    experiment_results.to_excel(f"experiment_result_{experiment_name}.xlsx")
    return experiment_results

    
    

if __name__ == '__main__':
    
    # gbest_vals, total_time = optimise(demand, supply)
    # plot_results(gbest_vals, total_time)

    print(experiment(split_particles_list[0:2], "trial"))