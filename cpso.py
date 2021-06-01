
import numpy as np
import pandas as pd
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
from pso import PSO, random_back, quantity, supply, feasible_vec, split_particles_list, experiment


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

    
def optimise(init_pos):

    start = time.perf_counter()

    iterations = 100
    
    gbest_val_list  = []
    gbest_pos_list  = []
    
    swarm = CPSO()
    # swarm.initialise()
    swarm.initialise_with_particle_list(init_pos)
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
            if feasible_vec(swarm.gbest_pos):
                print("Constraints met!")
    
    end = time.perf_counter()
    total_time = end-start

    return gbest_val_list, total_time



if __name__ == '__main__':
    
    # print(experiment(optimise, split_particles_list[:2], "test"))
    optimise(split_particles_list[0])