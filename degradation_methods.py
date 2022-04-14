# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:29:56 2022

@author: kamt_al
"""
import numpy as np

def exponential_degradation(health_index,row):
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
    
    return increment