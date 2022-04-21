# System Health Index Framework Guide

This notebook shows how to use of the System Health Index Framework.

## Set up of the Model
This task is performed in *model.py*.


```python
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from system_health_index_framework import Transfer,Composite,Health_Index
```

An aircraft model is initialized.


```python
aircraft1 = Composite(_id = "aircraft_01")
```

Various subsystems and components are added to the aircraft.


```python
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
```

Individual health indexes are added. In that case the degradation of the High Pressure Turbine Module. With the initailization of these health indexes a distribution for the start value needs to be given. Furthermore, a health state estimation method as well as a prediction method need to be defined. These methods are imported from the respective script.


```python
from degradation_methods import exponential_degradation
from health_state_estimation_method import method_exponential
```


```python
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
```

For demonstration purpose, history data is created for the aircraft. Real history data of an asset can be used instead.


```python
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

history = create_history_data(n_cycles=5,trajectory_type = "harsh",hour_mean=1.,hour_std=0.2)
```


```python
history
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time_utc</th>
      <th>hour</th>
      <th>cycle</th>
      <th>trajectory_type</th>
      <th>initial_wear</th>
      <th>a</th>
      <th>b</th>
      <th>noise_std</th>
      <th>excitation_energy_level</th>
      <th>cycle_abnormal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.650572e+09</td>
      <td>0.989507</td>
      <td>1</td>
      <td>harsh</td>
      <td>5.555556e-07</td>
      <td>0.001707</td>
      <td>1.414141</td>
      <td>0.001</td>
      <td>0.024738</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.650615e+09</td>
      <td>1.946683</td>
      <td>2</td>
      <td>harsh</td>
      <td>3.838384e-07</td>
      <td>0.001586</td>
      <td>1.438384</td>
      <td>0.001</td>
      <td>0.048667</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.650659e+09</td>
      <td>2.782158</td>
      <td>3</td>
      <td>harsh</td>
      <td>3.939394e-07</td>
      <td>0.001121</td>
      <td>1.412121</td>
      <td>0.001</td>
      <td>0.069554</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.650702e+09</td>
      <td>3.278484</td>
      <td>4</td>
      <td>harsh</td>
      <td>2.929293e-07</td>
      <td>0.002051</td>
      <td>1.555556</td>
      <td>0.001</td>
      <td>0.081962</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.650745e+09</td>
      <td>3.939805</td>
      <td>5</td>
      <td>harsh</td>
      <td>7.979798e-07</td>
      <td>0.001222</td>
      <td>1.531313</td>
      <td>0.001</td>
      <td>0.098495</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



In a first run, to propagate the introduced health indexes up to the parent aircraft, the health condition of the aircraft is updated.


```python
aircraft1.update_health_condition(utc_time = datetime.now().timestamp())

```

The history data is added to the aircraft model and the current health status is estimated.


```python
aircraft1.add_history_data(history)
```


```python
aircraft1.estimate_current_health_condition()
```

The model is saved.


```python
model = aircraft1
with open('./system_models/model.pickle', 'wb') as f:
     pickle.dump(model, f)
```

## Set up and Execution of the Simulation

Subsequently the generated model is used for simulation of the health index development of the system and its subsystems. This task is performed in *simulation.py*.


```python
from system_health_index_framework import Simulation, load_model
```

The generated model is loaded and a Simulation is initialized using the framework's *Simulation* class.


```python
#load created model
model,class_list = load_model("model")
Composite,Health_Index,Transfer,Schedule,Method = class_list

#initialize simulation
Simulation_01 = Simulation(simulation_id = "Simulation_01")
Simulation_01.model = model
```

Using the *instances* attribute of the classes from the class list above, a list of all class objects is shown.


```python
Composite.instances
```




    {'aircraft_01': <system_health_index_framework.Composite at 0x27164198748>,
     'engine_01': <system_health_index_framework.Composite at 0x271641bb148>,
     'engine_02': <system_health_index_framework.Composite at 0x271641bb648>,
     'hp_turbine1_01': <system_health_index_framework.Composite at 0x271641bb088>,
     'hp_turbine1_rotor_01': <system_health_index_framework.Composite at 0x271641bb488>,
     'hp_turbine1_disk_01': <system_health_index_framework.Composite at 0x271641bb4c8>,
     'hp_turbine2_01': <system_health_index_framework.Composite at 0x271641bb5c8>,
     'hp_turbine2_rotor_01': <system_health_index_framework.Composite at 0x271641bb548>,
     'hp_turbine2_disk_01': <system_health_index_framework.Composite at 0x271640d3b88>}



Simulation schedules are created. Each Schedule describes a possible health index development. 


```python
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
```


```python
#add schedules to simulation
history = model.history.table
schedules = create_simulation_schedules(history,N=10,n_cycles=100,trajectory_type = "harsh",hour_mean=1.,hour_std=0.2)
Simulation_01.add_schedules(schedules)
Simulation_01.run_simulation()
```

    Simulation S1
    Simulation S2
    Simulation S3
    Simulation S4
    Simulation S5
    Simulation S6
    Simulation S7
    Simulation S8
    Simulation S9
    Simulation S10
    


```python
from visualization_tools import visualize_simulation

```


```python
l_hi = ['hp_turbine1_01_hi_02','hp_turbine2_01_hi_02','aircraft_01_HI_00']
fig = visualize_simulation(Simulation_01,Health_Index,l_hi)
```


    


    
![png](output_32_1.png)
    

