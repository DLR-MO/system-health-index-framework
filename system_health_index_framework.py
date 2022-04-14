# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:17:04 2021

@author: kamt_al
"""
import abc
import gc
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

###############################
"""
USE CASES FOR HEALTH INDEX
impact of blade damage on parent is to be seen in context of other blades
    transfer method dependent on other health indices, transfer relative
each cycle of LLP consumes health index and single part has huge impact on parent
    transfer 1 to 1
health index transfer function needs to be defined for origin and destination
one or multiple hi can have a certain defined impact on target hi
"""


class Simulation:
    instances = dict()
    def __init__(self,simulation_id = "Simulation1"):
        self.__class__.instances[simulation_id] = self
        self.id = simulation_id
        self.model = None
        self.schedules = dict()
        
    def add_schedules(self,schedules):
        """
        add object of schedule class or list of objects
        """
        def add_elements(self,schedules):
            for elem in schedules:
                
                if elem.id in self.schedules.keys():
                    print("schedule -%s- already existing, please use other schedule id instead"%(elem.id))
                else:
                    self.schedules[elem.id] = elem
        try:
            
            add_elements(self,schedules)
            
        except:
            
            add_elements(self,[schedules])

    def run_simulation(self):
        """
        -run all schedules
        -analyze schedules, evaluate and save in model with simulation id
        """
        for schedule_id in self.schedules:
            print("Simulation",schedule_id)
            #pass simulation schedule to model
            self.model.run_degradation_simulation(Schedule.instances[schedule_id])
            
    def show_simulation_results(self):
        pass
class Method:
    instances = dict()
    def __init__(self,_id = "M1"):
        """
        should take an arbitrary number of schedules and its parameters 
        to link it to method function
        """
        self.__class__.instances[_id] = self
        self.id = _id
        #input params: {"schedule_id":["param1,param2,...."]}
        #independent variables: cycles, flight hours,...
        #dependent variables etc.
        self.input_parameters = dict()
        self.output_parameters = dict()
        self.method = 1
        
    def run(self):
        """
        run method with input parameters
        """
        print(self)

class Transfer:
    instances = dict()
    def __init__(self,prio = 1,transfer_method = np.min,transfer_id="Transfer1",source_id=[("S1","S1")],target_id=[("T1","T1")]):
        self.__class__.instances[transfer_id] = self
        self.source_id = source_id
        self.target_id = target_id
        self.id = transfer_id
        self.transfer_method = transfer_method
        self.prio = prio
        self.active = True
        #add Transfer to Health Index transfer relations
#    def __call__(self,inputs):
#        #take latest hi of target and update with latest source hi
#        return self.name + inputs

class Schedule:
    instances = dict()
    def __init__(self,_id="XXX"):#,name,generate_table_func):
        self.__class__.instances[_id] = self
        self.id = _id
        self.table = pd.DataFrame()
    
    def add_table(self,df):
        self.table = df
    
    def add_info(self,add_info_func):
        #add value column, define column name, define inputs from table if necessary
        #run method for column
        #add results of method to table in created column
        self.table = add_info_func(self.table)



class Health_Index:
    instances = dict()
    def __init__(self,_id="ID01",parent_id ="PID01",start_time=datetime.now().timestamp(),start_value = np.ones(1), start_value_probability = np.ones(1), prediction_method = None, health_state_method = None,ref_value = 1,ref_unit = ""):
        """
        id for hi itself
        parent_id for component affected
        value as dataframe with timestamp
        """
        self.__class__.instances[_id] = self
        self.parent_id = parent_id
        self.id = _id
        self.value = pd.DataFrame({"time_utc":[start_time],"value":[start_value],"value_probability":[start_value_probability],"simulation_flag":[False]})
        self.prediction_method = prediction_method
        self.health_state_method = health_state_method
        self.transfer_relations = dict()
        self.ref_value = ref_value
        self.ref_unit = ref_unit
    

class Component(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def operation(self):
        pass

class Composite(Component):
    instances = dict()
    def __init__(self,_id = "XXX" , part_no="XXX", serial_no = "XXX", typ = "arbitrary"):
        self.__class__.instances[_id] = self
        self.id = _id
        self.part_no = part_no
        self.part_serial = serial_no
        self._children = dict()
        self._parents = dict()
        self.health_indices = dict() #contains all health indexes of component and its children

        self.history = Schedule()
        self.transfer_relations = dict()
        
        #add automatically an overall component health index when component is created with default transfer relations
        self.health_indices["%s_HI_00"%(self.id)] = Health_Index(
                _id = "%s_HI_00"%(self.id),
                parent_id = self.id )
        #add default transfer relation to overall component HI, initially without source_ids
        #source ids are added if sub hi are added to component
        self.transfer_relations["hi_to_%s_HI"%(self.id)] = Transfer(
                transfer_method = np.min,
                transfer_id = "hi_to_%s_HI"%(self.id),
                source_id = [],
                target_id = [(self.id,"%s_HI_00"%(self.id))]
                )
        #add transfer_relation to HI transfer_relations overview
        Health_Index.instances["%s_HI_00"%(self.id)].transfer_relations["hi_to_%s_HI"%(self.id)] = Transfer.instances["hi_to_%s_HI"%(self.id)]
        #each health index is dependent on other health indices or parameters from schedule
        #there is always a method as transfer between params/target health index or source HI/target hi
        #there can be target hi already existing, or they have to be initialized if not
    
    def add_history_data(self,table):
        """
        add data frame to history data
        """

        self.history.table = self.history.table.append(table,sort = True)

    
    def operation(self):
        for child in self._children:
            child.operation()

    def add(self, component):
        def search_dict(dct,search_key,return_key=True):
            if return_key:
                return [key for key, val in dct.items() if search_key in key]
            else:
                return [val for key, val in dct.items() if search_key in key]
        #amend parents and children description
        self._children[component.id] = component
        component._parents[self.id] = self
#        self.health = pd.merge(self.health,component.health,
#                               how="outer",on="time",sort=True)
        #add main HI automatically to parent component: add Transfer relation
        #get current number of hi
        search_key = "%s_hi"%(self.id)
        hi_idx = len(search_dict(self.health_indices,search_key)) + 1
        if(len(str(hi_idx)) < 2):
            hi_idx = "0" + str(hi_idx)
        else:
            hi_idx = str(hi_idx)
        
        self.add_transfer_relation(
                Transfer(transfer_method = np.min,
                         transfer_id = "%s_HI_to_%s_hi_%s"%(component.id,self.id,hi_idx),
                         source_id = [("%s"%(component.id),"%s_HI_00"%(component.id))],
                         target_id = [("%s"%(self.id),"%s_hi_%s"%(self.id,hi_idx))]  )
                                                    )
        component.transfer_relations["%s_HI_to_%s_hi_%s"%(component.id,self.id,hi_idx)] = Transfer.instances["%s_HI_to_%s_hi_%s"%(component.id,self.id,hi_idx)]
    
    def add_health_index(self,health_index,source="normal"):
        """
        add object of health index class or list of objects
        """
        def add_elements(self,health_index):
            for elem in health_index:
                if elem.id in self.health_indices.keys():
                    print("health index -%s- already existing, please use other health index id instead"%(elem.id))
                else:
                    self.health_indices[elem.id] = elem
                    #for sub health index add transfer relation to component's overall Health Index
                    self.transfer_relations["hi_to_%s_HI"%(self.id)].source_id.append((self.id,elem.id))
                    #add to transfer relations dict of health index
                    elem.transfer_relations["hi_to_%s_HI"%(self.id)] = Transfer.instances["hi_to_%s_HI"%(self.id)]
                    
        try:
            add_elements(self,health_index)
        except TypeError:
            health_index = [health_index]
            add_elements(self,health_index)


    
    def add_transfer_relation(self,transfer_relation):
        """
        add object of transfer class or list of objects
        """
        def search_dict(dct,search_key,return_key=True):
            if return_key:
                return [key for key, val in dct.items() if search_key in key]
            else:
                return [val for key, val in dct.items() if search_key in key]
        
        def get_current_hi_index(component):
            search_key = "%s_hi"%(component.id)
            hi_idx = len(search_dict(component.health_indices,search_key)) + 1
            if(len(str(hi_idx)) < 2):
                hi_idx =  "0" + str(hi_idx)
            else:
                hi_idx = str(hi_idx)
            return "%s_hi_%s"%(component.id,hi_idx)
        
        def update_in_alist(alist, key, value, old_value):
            return [(k,v) if (v != old_value) else (key, value) for (k, v) in alist]
        
        def deactivate_transfer_relations(self,new_transfer_relation):
            """
            deactivate transfer relations with same source and lower prio
            """
            #create list of new transfer relation's source ids
            new_source_ids = [v for k,v in new_transfer_relation.source_id]
            
            for transfer_relation_key in self.transfer_relations.keys():
                transfer_relation = Transfer.instances[transfer_relation_key]
                source_ids = [v for k,v in transfer_relation.source_id]
                    
                if (any(x in source_ids for x in new_source_ids) & (transfer_relation.prio < new_transfer_relation.prio)):
                    transfer_relation.active = False
                else:
                    pass
            
        
        try:
            
            for elem in transfer_relation:
                if elem.id in self.transfer_relations.keys():
                    print("transfer id -%s- already existing, please use other transfer id instead"%(elem.id))
                    
                else:
                    self.transfer_relations[elem.id] = elem
#########################to function

                    updated_target_id = elem.target_id
                    for target_id in elem.target_id:
                        component_id,hi_id = target_id
                        component = Composite.instances[component_id]
                        if hi_id in component.health_indices:
                            #get current number of hi
                            old_hi_id = hi_id
                            hi_id = get_current_hi_index(component)
                            
                            #change Transfer object target id
                            updated_target_id = update_in_alist(elem.target_id,component.id,hi_id,old_hi_id)
                        else:
                            pass
                        
                        component.add_health_index(Health_Index(parent_id = component.id,_id = hi_id),source="tf")
                        #add transfer_relation to health index overview
                        health_index = Health_Index.instances[hi_id]
                        health_index.transfer_relations[elem.id] = elem
                    
                    #replace target_id list with updated list
                    elem.target_id = updated_target_id

                    for source_id in elem.source_id:
                        component_id,hi_id = source_id
                        component = Composite.instances[component_id]
#                        if hi_id in component.health_indices:
#                            #get current number of hi
#                            hi_id = get_current_hi_index(component)
#                        else:
#                            pass
#                        component.add_health_index(Health_Index(parent_id = component.id,_id = hi_id),source="tf")
                        #add transfer_relation to health index overview
                        health_index = Health_Index.instances[hi_id]
                        health_index.transfer_relations[elem.id] = elem
                #deactivate transfer relations with same source and lower prio
                #loop through all transfer relations of component
                deactivate_transfer_relations(self,elem)
##########################################################
        except TypeError:
            elem = transfer_relation
            if elem.id in self.transfer_relations.keys():
                print("transfer id -%s- already existing, please use other transfer id instead"%(elem.id))
                
            else:
                self.transfer_relations[elem.id] = elem
##########################################to function
                updated_target_id = elem.target_id
                for target_id in elem.target_id:
                        component_id,hi_id = target_id
                        component = Composite.instances[component_id]
                        if hi_id in component.health_indices:
                            #get current number of hi
                            old_hi_id = hi_id
                            hi_id = get_current_hi_index(component)
                            
                            updated_target_id = update_in_alist(elem.target_id,component.id,hi_id,old_hi_id)
                        else:
                            pass
                        component.add_health_index(Health_Index(parent_id = component.id,_id = hi_id),source="tf")
                        #add transfer_relation to health index overview
                        health_index = Health_Index.instances[hi_id]
                        health_index.transfer_relations[elem.id] = elem
                        #change health index id in target_id of transfer_relation
                #replace target_id list with updated list
                elem.target_id = updated_target_id
                

                
                for source_id in elem.source_id:
                        component_id,hi_id = source_id

                        component = Composite.instances[component_id]
#                        if hi_id in component.health_indices:
#                            #get current number of hi
#                            hi_id = get_current_hi_index(component)
#                        else:
#                            pass
#                        component.add_health_index(Health_Index(parent_id = component.id,_id = hi_id),source="tf")
                        
                        #add transfer_relation to health index overview
                        health_index = Health_Index.instances[hi_id]
                        health_index.transfer_relations[elem.id] = elem
                #deactivate transfer relations with same source and lower prio
                #loop through all transfer relations of component
                deactivate_transfer_relations(self,elem)
                
        ############################################################
        #a health index transfer function needs to be passed when component is added
        #component can have multiple effects, impact isolated or in a group e.g group of blades/single rotor disk
        #each part can have multiple health indices, each related to one degradation mechanism 
#degradation simulatoin takes schedule and updates health table
        #

    def estimate_current_health_condition(self):
        """
        run through all health index relations of parents and children 
        up to the defined point in time
        """
        def search_dict(dct,search_key,return_key=True):
            if return_key:
                return [key for key, val in dct.items() if search_key in key]
            else:
                return [val for key, val in dct.items() if search_key in key]
        
        for row in self.history.table.itertuples():
            
            #print(row.time)
            
            #update alle hi with defined method and append results to health index table
            list_of_children = [list(self._children.keys())]

            temp_list_of_children = []
            list_len = len(list_of_children[0])
            
            while (list_len > 0):
                temp_list_of_children = []
                list_len = len(temp_list_of_children)
                for elem in list_of_children[-1]:
                    temp_list_of_children.extend(list(Composite.instances[elem]._children.keys()))
                
                list_len = len(temp_list_of_children)
                
                if (list_len > 0):
                    #
                    list_of_children.append(temp_list_of_children)
                else:
                    pass
            #reverse list
    
            list_of_children = list_of_children[::-1]
    
            #flatten list and add current component to tail of list
            list_of_children = [y for x in list_of_children for y in x] + [self.id]
            
            #for each component update each hi which is not HI and only concerning the component itself
            for component in list_of_children:
                list_of_hi = search_dict(Composite.instances[component].health_indices,"",return_key = True)
                for elem in list_of_hi:
                    if Health_Index.instances[elem].health_state_method:

                        #calculate increment with prediction method for given input
                        value,value_probability = Health_Index.instances[elem].health_state_method(Health_Index.instances[elem],row)
                        #add new value as distribution, single value with probability 1.
                        Health_Index.instances[elem].value = Health_Index.instances[elem].value.append(
                                pd.DataFrame({"time_utc":[row.time_utc],
                                              "value":[value],
                                              "value_probability":[value_probability],
                                              "simulation_id":[""]}))

                    else:
                        pass

            
            #update all children and parent bottom up
            self.update_health_condition(utc_time = row.time_utc,simulation_id = "")
            
    def run_degradation_simulation(self,simulation_schedule):
        """
        run through all health index relations of parents and children 
        up to the defined point in time
        """
        def search_dict(dct,search_key,return_key=True):
            if return_key:
                return [key for key, val in dct.items() if search_key in key]
            else:
                return [val for key, val in dct.items() if search_key in key]
        
        for row in simulation_schedule.table.itertuples():
            
            #print(row.time)
            
            #update alle hi with defined method and append results to health index table
            list_of_children = [list(self._children.keys())]

            temp_list_of_children = []
            list_len = len(list_of_children[0])
            
            while (list_len > 0):
                temp_list_of_children = []
                list_len = len(temp_list_of_children)
                for elem in list_of_children[-1]:
                    temp_list_of_children.extend(list(Composite.instances[elem]._children.keys()))
                
                list_len = len(temp_list_of_children)
                
                if (list_len > 0):
                    #
                    list_of_children.append(temp_list_of_children)
                else:
                    pass
            #reverse list
    
            list_of_children = list_of_children[::-1]
    
            #flatten list and add current component to tail of list
            list_of_children = [y for x in list_of_children for y in x] + [self.id]
            
            #for each component update each hi which is not HI and only concerning the component itself
            for component in list_of_children:
                list_of_hi = search_dict(Composite.instances[component].health_indices,"",return_key = True)
                for elem in list_of_hi:
                    if Health_Index.instances[elem].prediction_method:

                        #draw hi value from prior distribution
                        #if any simulation id doesnt exist yet, use last not-simulated entry
                        #otherwise use last entry of simulation with same id
                        if((Health_Index.instances[elem].value.simulation_id == simulation_schedule.id).any()):
                            mask = Health_Index.instances[elem].value.simulation_id == simulation_schedule.id
                            hi_sample = np.random.choice(
                                    Health_Index.instances[elem].value["value"][mask].iloc[-1],
                                    1,
                                    p = Health_Index.instances[elem].value["value_probability"][mask].iloc[-1])[0]
                        else:
                            mask = Health_Index.instances[elem].value.simulation_id == ""
                            hi_sample = np.random.choice(
                                    Health_Index.instances[elem].value["value"][mask].iloc[-1],
                                    1,
                                    p = Health_Index.instances[elem].value["value_probability"][mask].iloc[-1])[0]
                        #calculate increment with prediction method for given input
                        incr = Health_Index.instances[elem].prediction_method(Health_Index.instances[elem],row)
                        
                        #add new value as distribution, single value with probability 1.
                        
                        Health_Index.instances[elem].value = Health_Index.instances[elem].value.append(
                                pd.DataFrame({"time_utc":[row.time_utc],
                                              "value":[np.array([hi_sample + incr])],
                                              "value_probability":[np.array([1.])],
                                              "simulation_id":[simulation_schedule.id]}))
                       
                    else:
                        pass

            
            #update all children and parent bottom up
            self.update_health_condition(utc_time = row.time_utc,simulation_id = simulation_schedule.id)
    
    
    def update_health_condition(self,utc_time = datetime.now().timestamp(),simulation_id = ""):
        """
        update current health status over all connected components, 
        according to hi and transfer relations
        
        get current utc time
        """

        def search_dict(dct,search_key,return_key=True):
            if return_key:
                return [key for key, val in dct.items() if search_key in key]
            else:
                return [val for key, val in dct.items() if search_key in key]
        
        def search_list(lst,search_key):
            return [item for item in lst if search_key in item]
        
        def search_list_of_tuples(lst,search_key,search_value_1 = True, return_value_1=True):
            if search_value_1:
                if return_value_1:
                    return [value_1 for value_1,value_2 in lst if search_key in value_1]
                else:
                    return [value_2 for value_1,value_2 in lst if search_key in value_1]
            else:
                if return_value_1:
                    return [value_1 for value_1,value_2 in lst if search_key in value_2]
                else:
                    return [value_2 for value_1,value_2 in lst if search_key in value_2]
        
        def update_health_index(transfer_relation,utc_time,simulation_id):
            
            #define target
            target_hi_id = transfer_relation.target_id[0][1]
            target_hi = Health_Index.instances[target_hi_id]
            
            #define number of iterations for sampling
            #only one sample for simulation
            if(simulation_id != ""):
                N=1
            else:
                N = 1000
            
            transfer_samples = np.array([])
            #define step-size for hi scala
            step_size = 0.001
            
            #check if any sources existent, otherwise append last entries again
            if any(transfer_relation.source_id):
                #create input array from source id
                list_of_hi = [hi_id for component_id,hi_id in transfer_relation.source_id]
                #print(any(transfer_relation.source_id),transfer_relation.target_id)
                
                
                for i in np.arange(N):
                    hi_input = np.array([])
    
                    for hi_id in list_of_hi:
                        #print(hi_id)#,Health_Index.instances[hi_id].value["value_probability"].iloc[-1])
                        #draw sample from distribution
                        #print(Health_Index.instances[hi_id].value["value"].iloc[-1],Health_Index.instances[hi_id].value["value_probability"].iloc[-1])
                        hi_sample = np.random.choice(
                                Health_Index.instances[hi_id].value["value"].iloc[-1],
                                1,
                                p = Health_Index.instances[hi_id].value["value_probability"].iloc[-1])
                        #hi sample to input array 
                        hi_input = np.hstack((hi_input,hi_sample))
        
                    #use transfer_method 
                    output_hi = transfer_relation.transfer_method(hi_input)
                    
                    #add output hi to transfer samples
                    transfer_samples = np.hstack((transfer_samples,output_hi))
                    
                    
                
                #create distribution from N transfer samples
                
                n_steps = int((np.max(transfer_samples) - np.min(transfer_samples))/step_size)
                
                if(n_steps > 0):
                    bins = np.linspace(np.min(transfer_samples),np.max(transfer_samples),n_steps)
                    #get distribution
    
                    counts,edges = np.histogram(transfer_samples,bins = bins)
                    probabilities = counts/np.sum(counts)
                    #exclude rightmost edge
                    
                    edges = edges[:-1]
                else:
                    #if unique values in transfer samples
                    edges = transfer_samples
                    probabilities = np.ones(edges.shape)/edges.shape[0]
                
            else:
                
                #if no source ids available, add last entries
                edges = target_hi.value.value.iloc[-1]
                probabilities = target_hi.value.value_probability.iloc[-1]
            #add output value to target hi 
            #if time stamps doesnt exist, add row
            #print(target_hi.id,len(target_hi.value),target_hi.value["time_utc"].iloc[-1], utc_time)
            if(target_hi.value["time_utc"].iloc[-1] == utc_time):
                
                pass
            else:
                target_hi.value = target_hi.value.append(pd.DataFrame({"time_utc":[utc_time],
                                                                       "value":[edges],
                                                                       "value_probability":[probabilities],
                                                                       "simulation_id":[simulation_id]}))


        
        #check for children down to lowest level, 
        #update process should start there
        #get list of children bottom up
        list_of_children = [list(self._children.keys())]

        temp_list_of_children = []
        list_len = len(list_of_children[0])
        
        while (list_len > 0):
            temp_list_of_children = []
            list_len = len(temp_list_of_children)
            for elem in list_of_children[-1]:
                temp_list_of_children.extend(list(Composite.instances[elem]._children.keys()))
            
            list_len = len(temp_list_of_children)
            
            if (list_len > 0):
                #
                list_of_children.append(temp_list_of_children)
            else:
                pass
        #reverse list

        list_of_children = list_of_children[::-1]

        #flatten list and add current component to tail of list
        list_of_children = [y for x in list_of_children for y in x] + [self.id]

        #for each component update each hi which is not HI and only concerning the component itself
        for component in list_of_children:
            #print(component)
            list_of_hi = search_dict(Composite.instances[component].health_indices,"",return_key = True)
            
            for elem in list_of_hi:

                flag = False
                for transfer_relation in search_dict(Health_Index.instances[elem].transfer_relations,"",return_key=False):
                    #component in target and source ids, no others allowed
                    if ((component in search_list_of_tuples(transfer_relation.target_id,component,search_value_1 = True,return_value_1=True))&(component in search_list_of_tuples(transfer_relation.source_id,component,search_value_1 = True,return_value_1=True))):
                        flag = True
                    else:
                        flag = False
                        break
                hi = Composite.instances[component].health_indices[elem]
                #if timestamp is unique, flag is True and not HI append row
                
                if(flag & (hi.value["time_utc"].iloc[-1] != utc_time)&("hi" in hi.id)):
                    
                    #add new row with timestamp and previous hi value 
                    #and value probability if timestamp new
                    hi.value = hi.value.append(pd.DataFrame({"time_utc":            [utc_time],
                                                             "value":               [hi.value["value"].iloc[-1]],
                                                             "value_probability":   [hi.value["value_probability"].iloc[-1]],
                                                             "simulation_id":     [simulation_id]}))
                    
                else:
                    pass
        
#        for component in list_of_children:
            list_of_hi = search_dict(Composite.instances[component].health_indices,"",return_key = True)
            for elem in list_of_hi:
            
                for transfer_relation in search_dict(Health_Index.instances[elem].transfer_relations,"",return_key=False):

                    if(("hi" in Health_Index.instances[elem].id) & (elem in search_list_of_tuples(transfer_relation.target_id,elem,search_value_1 = False,return_value_1=False)) & (transfer_relation.active is True)):
                        update_health_index(transfer_relation,utc_time,simulation_id)

                    else:
                        pass

                        
#        for component in list_of_children:
            list_of_hi = search_dict(Composite.instances[component].health_indices,"",return_key = True)
            for elem in list_of_hi:
                
                flag = False
                for transfer_relation in search_dict(Health_Index.instances[elem].transfer_relations,"",return_key=False):
                    
                    if(("HI" in Health_Index.instances[elem].id) & (elem in search_list_of_tuples(transfer_relation.target_id,elem,search_value_1 = False,return_value_1=False))):
                        flag = True
                        break
                    else:
                        pass
                if flag:
                    
                    update_health_index(transfer_relation,utc_time,simulation_id)
                else:
                    pass
                
        for component in list_of_children:
            list_of_hi = search_dict(Composite.instances[component].health_indices,"",return_key = True)
            for elem in list_of_hi:
                if "HI" in elem:
                    pass
                    #print(elem,Health_Index.instances[elem].value[["time_utc","value"]].iloc[-1])

def load_model(model):
    """
    loads model
    """
    with open('./system_models/%s.pickle'%(model),"rb") as f:
        loaded_obj = pickle.load(f)
    #for all classes generate class instance dictionaries
    class_list = [Composite,Health_Index,Transfer,Schedule,Method]
    
    for obj in gc.get_objects():
        for _class in class_list:
            if isinstance(obj, _class):
                _class.instances["%s"%(obj.id)] = obj
    
    return loaded_obj,class_list
