# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:22:38 2022

@author: kamt_al
"""

import numpy as np

def method(table):
    """
    takes pandas data frame and adds a new column which is generated from 
    existing columns in the table as input
    """
    return table

def add_excitation_energy_level(table):
    """
    adds excitation energy increment according to type of flight 
    total excitation energy normalized to [0,1] 
    

    Parameters
    ----------
    table : pandas dataframe

    Returns
    -------
    table : pandas dataframe

    """
    
    categories = {"harsh"  :0.025,
                  "medium" :0.017,
                  "low"    :0.011 }
    

    table["excitation_energy_level"] = 0.
    
    factor = categories[table.trajectory_type.iloc[0]]
    
    stage_length = np.diff(table["hour"].values)
    stage_length = np.hstack((table["hour"].values[0],stage_length))
    table["excitation_energy_level"] = factor * stage_length
    table["excitation_energy_level"] = table["excitation_energy_level"].cumsum()
    
    return table

def add_abnormal_cycle_count(table):
    if((table.excitation_energy_level > 1).any()):
        table["cycle_abnormal"] = table["cycle"] - table["cycle"].values[np.where(table["excitation_energy_level"].values < 1)][-1]
    else:
        table["cycle_abnormal"] = 0.
        
    return table