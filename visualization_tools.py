# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:45:17 2022

@author: kamt_al
"""

import numpy as np
import matplotlib.pyplot as plt
# Another utility for the legend
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc # For the legend
from matplotlib import cm

def getColorMap():
    x=1
    n=10001
    cmap = cm.viridis


    bottom = np.array([[0.,0.,0.,0.]])
    upper = np.flipud(cmap(np.linspace(1-x, 1, n)))

    colors = np.vstack((bottom, upper))
    
    tmap = mc.LinearSegmentedColormap.from_list('map_white', colors)
    return tmap

def visualize_simulation(Simulation,Health_Index,l_health_indexes):
    plt.rcParams.update({'font.size': 24})
    #get list of simulation ids
    l_schedule_ids = [key for key in Simulation.schedules]
    #get data from simulations and create 1 table for plots
    simulation_table_dict = dict()
    for hi in l_health_indexes:
        l_arrays = []
        for schedule_id in l_schedule_ids:
            
            df = Health_Index.instances[hi].value
            n_cycles = len(df[df.simulation_id == schedule_id])
            values = np.array([elem[0] for elem in df.value[df.simulation_id == schedule_id]])
            l_arrays += [values]
            
        
        arr = np.concatenate(l_arrays).reshape(len(l_schedule_ids),n_cycles)
        
        #create table with normalized np.histogram count per column for each cycle
        bins = np.linspace(0.,1.,101)
        
        l_row_ps = []
        #loop through cycles and get distribution for all simulation schedules
        for row in arr.T:
            row_clip = row
            row_clip[np.where(row < 0.)] = 0.
            counts,edges = np.histogram(row_clip,bins=bins)
            p = counts/np.sum(counts)
            
            l_row_ps += [p]
        
        p_arr = np.concatenate(l_row_ps).reshape(n_cycles,len(bins[:-1]))
        
        #filter cycles where all hi are zero, keep only first
        clip_len = len(np.where(np.all(p_arr[:,1:]==0.,axis=1))[0]) - 1

        if(clip_len > 0):
            p_arr = p_arr[:-clip_len,:]

            simulation_table_dict[hi] = {"bins":bins[:-1],
                                         "cycles":np.arange(n_cycles)[:-clip_len],
                                         "probs":p_arr.T}
        else:


            simulation_table_dict[hi] = {"bins":bins[:-1],
                                         "cycles":np.arange(n_cycles),
                                         "probs":p_arr.T}
        #get history data from health indexes
        
    
    
    
    #create mosaic plot
    fig = plt.figure(constrained_layout=True,figsize=(10,15))
    axs = fig.subplot_mosaic("""
    CC
    AB
    """,
    gridspec_kw={
        # set the height ratios between the rows
        "height_ratios": [2, 1],
    })

    
    l_axs = ["A","B","C"]
    
    l_titles = ["A","B","C"]
    for i,hi in enumerate(l_health_indexes):
        key = l_axs[i]
        ax = axs[key]
        
        xgrid = simulation_table_dict[hi]["cycles"]
        ygrid = simulation_table_dict[hi]["bins"]
        value = simulation_table_dict[hi]["probs"]
        
        ax.set_xlim(np.min(xgrid),np.max(xgrid)+10)
        ax.set_ylim(np.min(ygrid),np.max(ygrid))
        
        ax.set_title(l_titles[i])
        ax.set_xlabel("Cycles")
        ax.set_xticks(np.linspace(0.,100.,6))
        if not (key=="B"):
            ax.set_ylabel("Health Index")
        
        ax.set_yticks(np.linspace(0.,1.,6))
        if(key=="B"):
            ax.set_yticklabels([])
        
        ax.pcolormesh(xgrid, ygrid,value, cmap=getColorMap(), vmin=0., vmax=1.)
        ax.grid()

    
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.025])
    
    # Create a normalizer that goes from minimum to maximum temperature
    norm = mc.Normalize(0., 1.)

    # Create the colorbar and set it to horizontal
    cb = fig.colorbar(
    ScalarMappable(norm=norm, cmap=getColorMap()), 
    cax=cbar_ax, # Pass the new axis
    orientation = "horizontal"
                                        )

    # Remove tick marks
    cb.ax.xaxis.set_tick_params(size=0)

    # Set legend label
    cb.set_label("Probability Density")
    fig.savefig("./plots/simulation_plot.png",dpi=400)

    return fig

