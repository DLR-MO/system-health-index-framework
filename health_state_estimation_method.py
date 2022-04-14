# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:48:00 2022

@author: kamt_al
"""

import numpy as np

def method_exponential(health_index,row):
    #excitation energy threshold
    excitation_energy_level_threshold = 1.
    normal_degradation_slope = -0.001
    if(row.excitation_energy_level < excitation_energy_level_threshold):
        increment = normal_degradation_slope
    else:
        noise = np.random.normal(0.,row.noise_std)
        increment = - row.a * row.b * row.cycle_abnormal**(row.b - 1.) * np.exp(row.a*(row.cycle_abnormal)**(row.b))  + noise
    
    if(row.cycle == 1.):
        increment -= row.initial_wear
    else:
        pass
    
    #create noise around estimation and draw increment samples
    #
    N=1000
    mu,sigma = increment,0.001
    increment_samples = np.random.normal(mu,sigma,N)
    #draw samples from prior health state
    prior_samples = np.random.choice(health_index.value.value.iloc[-1],
                                     N,
                                     p=health_index.value.value_probability.iloc[-1])
    #calculate new health index samples
    samples = prior_samples + increment_samples
    #get new distribution of health index samples
    step_size = 0.001
    n_steps = int((np.max(samples) - np.min(samples))/step_size)
    bins = np.linspace(np.min(samples),np.max(samples),n_steps)
    #get distribution
    
    counts,edges = np.histogram(samples,bins = bins)
    probabilities = counts/np.sum(counts)
    #exclude rightmost edge
                    
    edges = edges[:-1]
    return edges,probabilities
