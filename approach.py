# coding: utf-8

import numpy as np
from numpy import linalg as LA
import gym
import copy
from itertools import cycle
from functools import partial


class Reservoir():
    def __init__(self, in_size, hid_size, leaking_rate):
        self.in_size = in_size
        self.hid_size = hid_size
        self.leaking_rate = leaking_rate
        
        # TODO: maybe add the bias instead of concatenation 
        self.in_weights = np.random.normal(0, 1, [hid_size, in_size+1])
        self.hid_weights = np.random.normal(0, 1, [hid_size, hid_size])
        self.prev_state = np.random.normal(0,1,[hid_size])
        
    def forward(self, inpt):
        update = np.tanh(self.in_weights @ np.append(inpt,1) + self.hid_weights @ self.prev_state)
        
        self.prev_state = (1 - self.leaking_rate) * self.prev_state + leaking_rate * update
        
        return self.prev_state
        
    
    def _init_weights(self, desired_spectral_radius):
        # TODO: make the spectral radius == 1
        
        #self.in_weights = np.random.uniform(, , [hid_size, in_size+1])
        self.in_weights = np.random.normal(0, 1, [hid_size, in_size+1])
        
        #self.hid_weights = np.random.uniform(, , [hid_size, in_size+1])
        self.hid_weights = np.random.normal(0, 1, [hid_size, hid_size])
        hid_spectral_radius = np.abs(LA.eigvals(self.hid_weights)).max()
        s_radius_scaling_factor = desired_spectral_radius/hid_spectral_radius
        self.hid_weights *= s_radius_scaling_factor
        
        self.prev_state = np.random.normal(0,1,[hid_size])
        
    


# In[291]:


class Readout():
    def __init__(self, input_size, hidden_size, output_size):
        self.weights = np.random.uniform(0, 1, size=[output_size, input_size + hidden_size + 1])
        
        self.shape = [output_size, input_size + hidden_size + 1]
        self.N = np.random.uniform(0,1, self.shape)
        self.A = np.random.uniform(0,1, self.shape)
        self.B = np.random.uniform(0,1, self.shape)
        self.C = np.random.uniform(0,1, self.shape)
        self.D = np.random.uniform(0,1, self.shape)
        
        self.prev_input = None
        self.prev_output = None
        
    def forward(self, inpt, reservoir_activations):
        self.prev_input = np.concatenate([inpt, reservoir_activations, [1]])
        self.prev_output = np.tanh(self.weights @ self.prev_input)
        
        self.update()
        
        return self.prev_output

    def weight_references(self):
        return [self.N, self.A, self.B, self.C, self.D]

    def get_weights(self):
        return np.array([self.N, self.A, self.B, self.C, self.D])
    
    def forward_no_update(self, inpt, reservoir_activations):
        return np.tanh(self.weights @ np.concatenate([inpt, reservoir_activations, [1]]))
    
    def update(self):
        correlation =  np.multiply(self.A, np.outer(self.prev_input, self.prev_output).T)
        
        # TODO: check if the numpy expands the multiplication correctly
        pre_synaptic = np.multiply(self.B, self.prev_input)
        # TODO: check if the numpy expands the multiplication correctly
        post_synaptic = np.multiply(self.C, self.prev_output.reshape(-1,1))

        self.weights += np.multiply(self.N, correlation + pre_synaptic + post_synaptic)


# $\Delta w_{ij} = \eta \cdot \left( \mathrm{A}o_i o_j  + \mathrm{B}o_i + \mathrm{C}o_j + \mathrm{D} \right)$

# In[292]:


class Sigmoid_layer():
    def __init__(self, input_size):
        #self.l = np.random.normal(0.2, 0.05, size=[input_size])*20
        self.l = np.array([2.2, 1.3, 23, 1.3])

        
    def forward(self, x):
        return 2*(1/(1 + np.exp(-self.l*x))-1/2)


# In[301]:




def model(sigmoid, reservoir, readout, inpt):
    return readout.forward(inpt,reservoir.forward(sigmoid.forward(inpt)))

def model_no_update(sigmoid, reservoir, readout, inpt):
    return readout.forward_no_update(inpt,reservoir.forward(sigmoid.forward(inpt)))

#population = [partial(model, sigmoids[i], reservoir, readout) for i,readout in enumerate(readouts)]


# $\mathrm{h_{t+1}} = \mathrm{h_t} + \frac{\alpha}{n\sigma} \sum_{i=1}^{n} F\left(\mathrm{h_t}+ \sigma\epsilon_i\right) \cdot \epsilon_i $

# In[302]:



def run_model(f):
    initial_state=env.reset()
    appendedObservations=[]

    action = f(np.zeros(4))
    
    total_reward = 0
    
    for timeIndex in range(timeSteps):
        
        observation, reward, terminated, truncated, info = env.step(int(action[0]>0))
        
        total_reward += reward
        
        action = f(observation)
        
        appendedObservations.append(observation)
        if (terminated):
            break
            
    return total_reward    
    

def test_population(sigmoids, readouts, n_iterations):
    rewards = []
    scores = []

    for idx, readout in enumerate(readouts):
        
        f_without_updates = partial(model_no_update, sigmoids[idx], reservoir, readout)
        f_updated = partial(model, sigmoids[idx], reservoir, readout)
        
        rs = []
        tmp = []
        for _ in range(n_iterations):

            r1 = run_model(f_without_updates)
            r2 = run_model(f_updated)
            r3 = run_model(f_without_updates)

            tmp.append([r1,r2,r3])
            rs.append(r3-r1)    

        scores.append(tmp)
        rewards.append(sum(rs)/n_iterations)    


    return rewards, scores


def crossover(p1,p2):
    parent1, parent2 = p1.flatten(), p2.flatten()

    shape = parent1.shape

    crossover_point = np.random.randint(len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child.reshape(shape)

def crossover_individual(a, b):

    offspring = copy.deepcopy(a)
    b_weights = b.get_weights()

    weights = offspring.weight_references()
    for i, a_weights in enumerate(a.get_weights()):
        weights[i] = crossover(a_weights,b_weights[i])

    return offspring


def crossover_population(population, idcs):
    pairs = zip(idcs, idcs[::-1])
    offsprings = []
    for i,j in pairs:
        offsprings.append(crossover_individual(population[i],population[j]))

    return offsprings


def mutate(dna, ratio_denominator, sigma, replace = False):

    # create indeces to select random elements
    ln = len(dna) 
    idcs = np.arange(ln)

    # create a normally distributed noise
    noise = np.random.normal(scale = sigma,size=ln)

    # shuffle to random uniform select
    np.random.shuffle(idcs)

    slice_point = ln // ratio_denominator

    selection_idcs = idcs[:slice_point]

    # check either to select or add the noise
    if replace:
        dna[selection_idcs] = noise[:slice_point]
    else:
        dna[selection_idcs] += noise[:slice_point]
    
    return dna

def mutate_population(population, ratio_denominator, idcs, sigma, replace):
    mutants = []

    for i in idcs:
        mutant = copy.deepcopy(population[i])
        for weights in mutant.weight_references():
            shape = weights.shape
            weights = mutate(weights.flatten(), ratio_denominator, sigma, replace = replace).reshape(shape)
        mutants.append(mutant)

    return mutants

    
if __name__ == "__main__":
    population_size = 1000
    output_size = 1
    input_size = 4
    hidden_size = 1000
    leaking_rate = 2

    reservoir = Reservoir(input_size, hidden_size, leaking_rate)

    readouts = [Readout(input_size, hidden_size, output_size) for i in range(population_size)]

    sigmoids = [Sigmoid_layer(input_size) for i in range(population_size)]

    env = gym.make("CartPole-v1", render_mode='rgb_array')

    n_episodes = 10
    timeSteps=8000
    n_iterations = 10

    # simulate the environment
    for idx in range(n_episodes):
        rewards, scores = test_population(sigmoids, readouts, n_iterations)
        print(idx, ":", np.mean(rewards))
        print(scores)

        ranks = np.argsort(rewards)[::-1]

        mutation_ratio_denominator = 3

        # select the best

        n_best = len(readouts) // 3

        selected_idcs = ranks[:n_best]

        n_missing = population_size - 2 * n_best

        offsprings = crossover_population(readouts, selected_idcs)

        pool = cycle(ranks[:n_best])

        mutant_idcs = [next(pool) for i in range(n_missing)]

        mutants = mutate_population(readouts, mutation_ratio_denominator, mutant_idcs, 1, replace = False)

        readouts = [readouts[i] for i in ranks[:n_best]] + offsprings + mutants


    rewards, scores = test_population(sigmoids, readouts, n_iterations)
    print("Final:", np.mean(rewards))
    print(scores)




env.close()
