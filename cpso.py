
import numpy as np
import pandas as pd
from dataclasses import dataclass
import time
import matplotlib.pyplot as plt
from pso import PSO, demand, supply, feasible_vec, split_particles_list, experiment, time_function, poss_val, random_val, plot_results
from numba import jit, njit



## Funtions

@jit(nopython=True)
def random_back(position, velocity):
    ''' Takes a position and a velocity and returns a new position that
        meets demand & supply constraints '''
    global demand, supply
        
    vec = position + velocity
    if feasible_vec(vec):
        return vec
    else:
        inds = np.where(demand.flatten()!=0)[0] # Returns a tuple -> (inds, )
        new_pos = np.zeros(position.size)
        for i in inds:
            if poss_val(i, (int(position[i]+velocity[i])), new_pos):
                new_pos[i] = int(position[i] + velocity[i])
            else:
                r = random_val(new_pos, i)
                new_pos[i] = r
        
        if feasible_vec(new_pos):
            return new_pos
        else:
            print(f'random_back() returned an unfeasible vector')



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
            
        

@time_function    
def optimise(init_pos):

    start = time.perf_counter()

    iterations = 300
    
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
    
    # experiment(optimise, split_particles_list[45:50], "cpso[45_50]")
    # print("Done")
    gbest_vals, total_time = optimise(split_particles_list[0])
    plot_results(gbest_vals, total_time)