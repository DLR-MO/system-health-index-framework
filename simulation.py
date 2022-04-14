import pickle
import os
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from visualization_tools import visualize_simulation
from system_health_index_framework import Transfer,Composite,Health_Index,Simulation,Schedule,load_model

"""
create model/load model
get current health status/load current health status

create/load simulation schedules (scheduling tool?)
each row should have a discrete event take off, climb, cruise, descent and 
according parameters
scheduler uses distributions for creating these schedules
dependent probability distributions necessary

run simulation

save simulation results
visualize simulation results for choosen health indices
    (system visualization)
    ridgeline plot
    time series histogram

"""
def create_simulation_schedules(history,N=100,n_cycles=2,trajectory_type = "harsh",hour_mean=1.,hour_std=0.2):
    """
    creates schedules for simulation
    """
    from schedule_methods import add_excitation_energy_level,add_abnormal_cycle_count
    
    #read N-CMAPSS trajectory statistics
    df_stats = pd.read_csv("./data/trajectory_stats_N-CMAPSS_DS01-005.csv")
    df_stats = df_stats[df_stats.trajectory_type == trajectory_type]
    
    l_schedules = []
    
    for i,sim_id in enumerate(np.arange(N)+1):
        Simulation = Schedule(_id = "S%i"%(int(sim_id)))
        #add a value from random uniform distribution
        initial_wear = np.random.choice(np.linspace(0.,0.000001,100),n_cycles,p=np.ones(100)/100.)
        a = np.random.choice(np.linspace(0.001,0.003,100),n_cycles,p=np.ones(100)/100.)
        b = np.random.choice(np.linspace(1.4,1.6,100),n_cycles,p=np.ones(100)/100.)
        noise_std = np.zeros(n_cycles)
        noise_std[:] = 0.001
        
        #hour = np.random.choice(df_stats["hour"].values,n_cycles,p=np.ones(len(df_stats))/len(df_stats)).cumsum()
        hour = np.random.normal(hour_mean,hour_std,n_cycles).cumsum()
        cycle = np.arange(n_cycles) + 1
        
        trajectory_type_arr = np.ones(n_cycles).astype(object)
        trajectory_type_arr[:] = trajectory_type
        #add time_utc
        
        timedelta_utc = np.full(n_cycles,np.timedelta64(12, 'h')).cumsum()
        timedelta_utc_ux = timedelta_utc.astype(np.timedelta64) / np.timedelta64(1, 's')
        
        Simulation.add_table(pd.DataFrame({"time_utc":history.time_utc.values[-1] + timedelta_utc_ux,
                                           "hour":history.hour.values[-1] + hour,
                                           "cycle":history.cycle.values[-1] + cycle,
                                           "trajectory_type":trajectory_type_arr,
                                           "initial_wear":initial_wear,
                                           "a":a,
                                           "b":b,
                                           "noise_std":noise_std}))
        
        Simulation.add_info(add_excitation_energy_level)
        #add precollected ecitation energy amounts
        Simulation.table["excitation_energy_level"] + history["excitation_energy_level"].values[-1]
        
        Simulation.add_info(add_abnormal_cycle_count)
        
        l_schedules += [Simulation]

    
    return l_schedules

#load created model
model,class_list = load_model("model")

Composite,Health_Index,Transfer,Schedule,Method = class_list

#initialize simulation
Simulation_01 = Simulation(simulation_id = "Simulation_01")
Simulation_01.model = model

#add schedules to simulation
history = model.history.table
schedules = create_simulation_schedules(history,N=100,n_cycles=100,trajectory_type = "harsh",hour_mean=1.,hour_std=0.2)
Simulation_01.add_schedules(schedules)
Simulation_01.run_simulation()

l_hi = ['hp_turbine1_01_hi_02','hp_turbine2_01_hi_02','aircraft_01_HI_00']

visualize_simulation(Simulation_01,Health_Index,l_hi)

