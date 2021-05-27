
import numpy as np
import pandas as pd
from dataclasses import dataclass, astuple, field
import random
import time
import matplotlib.pyplot as plt
import cProfile, pstats, io # Needed for profile
import concurrent.futures


### Read data ####

xls = pd.ExcelFile('Data.xlsx')
products = pd.read_excel(xls, 'products', index_col=0, nrows=78, usecols='A:G')
customers = pd.read_excel(xls, 'customers', index_col=0, usecols='A:C')
quantity = pd.read_excel(xls, 'demand_boxes', index_col=0, nrows=78, usecols='A:CE').fillna(0).astype(np.int64)
prices = pd.read_excel(xls, 'price_per_box', index_col=0, nrows=78,usecols='A:CE')
transport = pd.read_excel(xls, 'transport_lookup', usecols='D:CG', skiprows=2, names=quantity.columns)
transport.set_index(quantity.index, inplace=True)
supply = products['Qty(eggs)']/products['eggs/pack']
cost_eggs = np.sum(products['Qty(eggs)'].values * products['Cost/egg'].values)

### Functions ###

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


def calculate_profit(vec):
    ''' Takes a vector and checks that it meets the constraints
        Reshapes the vector to match the shape of quantity matrix
        Sales is calculated by multyplying the qty values by prices
        Transport = qty values into transport cost lookup
        Cost of eggs = Sum of cost/eggs into total eggs for product
    '''
    global quantity, prices, transport, products
    qty = vec.reshape(quantity.shape)
    sales = np.sum(qty*prices.values)
    transport_cost = np.sum(transport.values * qty)
    cost_eggs = np.sum(products['Qty(eggs)'].values * products['Cost/egg'].values)
    total_costs = transport_cost + cost_eggs
    profit = sales-total_costs
    return np.round(profit)


def feasible_vec(vec, demand, supply):
    ''' Returns True if the vector meets demand and supply
        constraints'''
    mat = vec.reshape(demand.shape) 
    demand_check = np.all((mat <= demand) & (mat >=0))
    supply_check = np.all((mat.sum(axis=1)<=supply) & (mat.sum(axis=1)>=0))
    return demand_check and supply_check 


def poss_val(index, val, vec, demand, supply):
    ''' Returns True if the 'val' being placed in 
        'index' position of 'vec' meets 'demand' and 'supply' 
        constraints '''
    vec_copy = vec.copy()
    vec_copy[index]=val
    return feasible_vec(vec_copy, demand, supply)  


def random_val(vec, index, demand, supply):
    ''' Returns a random value for a vec at a given index position
        that meets demand and supply constraints'''
    mat = vec.reshape(demand.shape)
    mat_row_ind, mat_col_ind = np.unravel_index(index,(demand.shape))
    alloc_supply = [np.sum(m, axis=0) for m in mat]
    avail_supply = supply[mat_row_ind]-(alloc_supply[mat_row_ind]-mat[mat_row_ind][mat_col_ind])
    avail_demand = demand[mat_row_ind][mat_col_ind]
    if min(avail_supply, avail_demand)>0:
        return random.randint(0, min(avail_supply, avail_demand))
    else:
        return 0


def random_back(position, velocity, demand, supply):
    ''' Takes a position and a velocity and returns a new position that
        meets demand & supply constraints '''
    
    vec = position + velocity
    if feasible_vec(vec, demand, supply):
        return vec
    else:
        inds = np.where(demand.flatten()!=0)[0] # Returns a tuple -> (inds, )
        new_pos = np.zeros(position.size)
        for i in inds:
            if poss_val(i, (int(position[i]+velocity[i])), new_pos, demand, supply):
                new_pos[i] = int(position[i] + velocity[i])
            else:
                r = random_val(new_pos, i, demand, supply)
                new_pos[i] = r
        
        if feasible_vec(new_pos, demand, supply):
            return new_pos
        else:
            print(f'random_back() returned an unfeasible vector')


def random_back(position, velocity, demand, supply):
    ''' Takes a position and a velocity and returns a new position that
        meets demand & supply constraints '''
    
    vec = position + velocity
    if feasible_vec(vec, demand, supply):
        return vec
    else:
        inds = np.where(demand.flatten()!=0)[0] # Returns a tuple -> (inds, )
        new_pos = np.zeros(position.size)
        for i in inds:
            if poss_val(i, (int(position[i]+velocity[i])), new_pos, demand, supply):
                new_pos[i] = int(position[i] + velocity[i])
            else:
                r = random_val(new_pos, i, demand, supply)
                new_pos[i] = r
        
        if feasible_vec(new_pos, demand, supply):
            return new_pos
        else:
            print(f'random_back() returned an unfeasible vector')

def time_function(function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        end = time.perf_counter()
        print(f'\n{function.__name__}() took {round((end-start),2)} seconds to run')
        return result
    return wrapper


def profile(fnc):
    """ A decorator that uses cProfile to profile a function
        Source: Sebastiaan Mathôt https://osf.io/upav8/
    """
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


# def plot_results(gbest_vals):
#     x_axis_vals = [x for x in range(len(gbest_vals))]
#     plt.plot(x_axis_vals, gbest_vals)
#     plt.tight_layout()
#     plt.xlabel("Iterations")
#     plt.ylabel("Profit")
#     plt.show()

def plot_results(gbest_vals, total_time):
    _, ax = plt.subplots() 
    x_axis_vals = [x for x in range(len(gbest_vals))]
    ax.plot(x_axis_vals, gbest_vals)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Profit")
    props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    props2 = dict(boxstyle='round', facecolor='thistle', alpha=0.5)
    ax.text(0.1, 0.9, 'last gbest_val: '+str(round(gbest_vals[-1],2)), transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props1)
    ax.text(0.1, 0.7, 'total_time: '+str(round(total_time/60,2))+' minutes' , transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=props2)
    # ax.set_xlim([0,gbest_vals.size])
    # ax.set_ylim([min(gbest_vals)-10, 10+max(gbest_vals)])
    ax. ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.show()


def split_list(particle_list, num_particles):
    ''' Takes a list of particles and splits it by 
        the num_particles -> list'''
    return [particle_list[i:i+num_particles] for i in range(0, len(particle_list), num_particles)]


@dataclass
class CPSO:
    num_particles : int
    particles: list = field(default_factory=list)
    gbest_val: float = -np.Inf
    gbest_pos: np.ndarray = np.empty(quantity.values.flatten().size)


    def __post_init__(self):
        self.particles = [dict() for _ in range(self.num_particles)]
    
    
    def initialise(self, demand, supply):
        for particle in self.particles:
            particle['position'] = random_instantiates_vector(demand, supply)
            particle['pbest_val'] = -np.Inf
            particle['velocity'] = np.zeros(particle['position'].size)
    

    def pick_informants_ring_topology(self):
        for index, particle in enumerate(self.particles):
            particle['informants'] = []
            particle['lbest_val'] = -np.Inf
            particle['informants'].append(self.particles[(index-1) % len(self.particles)])
            particle['informants'].append(self.particles[index])
            particle['informants'].append(self.particles[(index+1) % len(self.particles)])
            

    def calculate_fitness(self):
        for particle in self.particles:
            particle['profit'] = calculate_profit(particle['position'])


    def set_pbest(self):
        for particle in self.particles:
            if particle['profit'] > particle['pbest_val']:
                particle['pbest_val'] = particle['profit']
                particle['pbest_pos'] = particle['position']
    

    def set_lbest(self):
        for particle in self.particles:
            for informant in particle['informants']:
                if(informant['pbest_val'] >= particle['lbest_val']):
                    informant['lbest_val'] = particle['pbest_val']
                    informant['lbest_pos'] = particle['pbest_pos']
    
    
    def set_gbest(self):
        for particle in self.particles:
            if particle['lbest_val'] >= self.gbest_val:
                self.gbest_val = particle['lbest_val']
                self.gbest_pos = particle['lbest_pos']
    

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

    
    def move_random_back(self, demand, supply):
        for particle in self.particles:
            new_pos = random_back(particle['position'], particle['velocity'], demand, supply)
            particle['position'] = np.floor(new_pos)

    
    
def optimise(demand, supply):
    
    start = time.perf_counter()

    iterations = 500
    
    gbest_val_list  = []
    gbest_pos_list  = []
    
    swarm = CPSO(20)
    swarm.initialise(demand, supply)
    swarm.pick_informants_ring_topology()
    
    for i in range(iterations):
        swarm.calculate_fitness()
        swarm.set_pbest()
        swarm.set_lbest()
        swarm.set_gbest()  
        swarm.set_constricted_velocity()
        swarm.move_random_back(demand, supply)

        print(f"Iteration: {i} gbest_val: {round(swarm.gbest_val, 2)}")    

        gbest_val_list.append(round(swarm.gbest_val, 2))
        if i == iterations-1: # Get the value from the last iteration
            gbest_pos_list.append(swarm.gbest_pos)
            if feasible_vec(swarm.gbest_pos, demand, supply):
                print("Constraints met!")
    
    end = time.perf_counter()
    total_time = round((end-start),2)

    return np.array(gbest_val_list), total_time

def experiment():
    
    num_runs = 100

    




if __name__ == '__main__':
    gbest_vals, total_time = optimise(quantity.values, supply.values)
    plot_results(gbest_vals, total_time)
    
