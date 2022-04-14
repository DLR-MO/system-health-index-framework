# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:22:11 2022

@author: kamt_al
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from system_health_index_framework import Transfer,Composite,Health_Index
from degradation_methods import exponential_degradation
from health_state_estimation_method import method_exponential

def create_history_data(n_cycles=10,trajectory_type = "harsh",hour_mean=1.,hour_std=0.2):
    from schedule_methods import add_excitation_energy_level,add_abnormal_cycle_count
    df_stats = pd.read_csv("./data/trajectory_stats_N-CMAPSS_DS01-005.csv")
    df_stats = df_stats[df_stats.trajectory_type == trajectory_type]
    initial_wear = np.random.choice(np.linspace(0.,0.000001,100),n_cycles,p=np.ones(100)/100.)
    a = np.random.choice(np.linspace(0.001,0.003,100),n_cycles,p=np.ones(100)/100.)
    b = np.random.choice(np.linspace(1.4,1.6,100),n_cycles,p=np.ones(100)/100.)
    noise_std = np.zeros(n_cycles)
    noise_std[:] = 0.001
    
    hour = np.random.normal(hour_mean,hour_std,n_cycles).cumsum()
    cycle = np.arange(n_cycles) + 1
        
    trajectory_type_arr = np.ones(n_cycles).astype(object)
    trajectory_type_arr[:] = trajectory_type
    #add time_utc
    time_utc = np.full(n_cycles,np.datetime64('now')) + np.full(n_cycles,np.timedelta64(12, 'h')).cumsum()
    time_utc_ux = time_utc.astype(np.timedelta64) / np.timedelta64(1, 's')
        
    df = pd.DataFrame({"time_utc":time_utc_ux,
                       "hour":hour,
                       "cycle":cycle,
                       "trajectory_type":trajectory_type_arr,
                       "initial_wear":initial_wear,
                       "a":a,
                       "b":b,
                       "noise_std":noise_std})
    
    df = add_excitation_energy_level(df)
    df = add_abnormal_cycle_count(df)

    return df

aircraft1 = Composite(_id = "aircraft_01")
    
engine1 = Composite(_id = "engine_01")
engine2 = Composite(_id = "engine_02")
    
hp_turbine1 = Composite(_id = "hp_turbine1_01")
hp_turbine1_rotor1 = Composite(_id = "hp_turbine1_rotor_01")
hp_turbine1_disk1 = Composite(_id = "hp_turbine1_disk_01")


hp_turbine2 = Composite(_id = "hp_turbine2_01")
hp_turbine2_rotor1 = Composite(_id = "hp_turbine2_rotor_01")
hp_turbine2_disk1 = Composite(_id = "hp_turbine2_disk_01")

hp_turbine1_rotor1.add(hp_turbine1_disk1)
hp_turbine1.add(hp_turbine1_rotor1)
engine1.add(hp_turbine1)

hp_turbine2_rotor1.add(hp_turbine1_disk1)
hp_turbine2.add(hp_turbine2_rotor1)
engine2.add(hp_turbine2)

aircraft1.add(engine1)
aircraft1.add(engine2)

#add health indices to model


hp_turbine1_dm1_hi_01 = Health_Index(_id = "hp_turbine1_01_hi_02",
                                            parent_id = "hp_turbine1_01",
                                            start_value = np.linspace(.9,1.,100),
                                            start_value_probability = np.ones(100)/100.,
                                            ref_value = 50.,
                                            ref_unit = "Kelvin",
                                            prediction_method = exponential_degradation,
                                            health_state_method = method_exponential)

hp_turbine2_dm1_hi_01 = Health_Index(_id = "hp_turbine2_01_hi_02",
                                            parent_id = "hp_turbine2_01",
                                            start_value = np.linspace(.9,1.,100),
                                            start_value_probability = np.ones(100)/100.,
                                            ref_value = 50.,
                                            ref_unit = "Kelvin",
                                            prediction_method = exponential_degradation,
                                            health_state_method = method_exponential)



hp_turbine1.add_health_index(hp_turbine1_dm1_hi_01)
hp_turbine2.add_health_index(hp_turbine2_dm1_hi_01)

#add history data
history = create_history_data(n_cycles=20,trajectory_type = "harsh",hour_mean=1.,hour_std=0.2)
#update current health status
aircraft1.update_health_condition(utc_time = datetime.now().timestamp())

# #estimate current health condition according to current
aircraft1.add_history_data(history)
aircraft1.estimate_current_health_condition()
#     #######save system model######
model = aircraft1
with open('./system_models/model.pickle', 'wb') as f:
     pickle.dump(model, f)