#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import pickle
import shutil 
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math
from IPython import embed


# In[14]:


# set up simulation constants

resolution =30

n_air = 1.0000003
n_Si = 1.45
n_SiO2 = 3.48

a = 0.5   # lattice period 

#pml_thickness = round(wavelength / 2,3)
pml_thickness = 0.780
height_hole = 5
height_SiO2 = height_hole
height_air = 7+2*pml_thickness
height_Si_initial= 0.5*(height_air-height_hole)
height_Si_add = 0.75*0.5*height_SiO2


cell_x = a
cell_y = a
cell_z = height_air

z_air = height_air

center_air = 0 
center_SiO2 = 0 
center_hole = 0
center_Si_initial = -0.5*height_hole -0.25*(height_air-height_hole)+0.005
center_Si_add = 0.5*height_Si_add


# In[15]:


#initial_geometry setup

cell_size = mp.Vector3(cell_x,cell_y,cell_z)
pml_layers = [mp.PML(thickness = pml_thickness, direction = mp.Z)]

geometry = [mp.Block(size=mp.Vector3(mp.inf,mp.inf,height_air), 
                    center=mp.Vector3(0,0,center_air),
                    material=mp.Medium(index=n_air)),
           mp.Block(size=mp.Vector3(mp.inf,mp.inf,height_Si_initial),
                   center=mp.Vector3(0,0,center_Si_initial),
                   material=mp.Medium(index=n_Si))]


k_point = mp.Vector3(0,0,0)

## source ##
center_source = -0.5*height_hole-0.125*(height_air-height_hole) 
source_cmpt = mp.Ey

wavelength = 1.55
freq = 1 / wavelength


sources = [mp.Source(mp.ContinuousSource(frequency=freq),
                    component=source_cmpt,
                    center=mp.Vector3(0,0,center_source),
                    size=mp.Vector3(cell_x,cell_y,0))]


## flux monitor ##
nfreq = 1
df = 0
fr_center = 0.5*height_hole+0.125*(height_air-height_hole)
fr = mp.FluxRegion(center=mp.Vector3(0,0,fr_center), 
            size=mp.Vector3(cell_x, cell_y, 0))



if source_cmpt == mp.Ey:
    symmetries = [mp.Mirror(mp.X, phase=+1), #epsilon has mirror symmetry in x and y, phase doesn't matter
                  mp.Mirror(mp.Y, phase=-1)] #but sources have -1 phase when reflected normal to their direction
elif src_cmpt == mp.Ex:                      #use of symmetries important here, significantly speeds up sim
    symmetries = [mp.Mirror(mp.X, phase=-1),
                  mp.Mirror(mp.Y, phase=+1)]
elif src_cmpt == mp.Ez:
    symmetries = [mp.Mirror(mp.X, phase=+1),
                  mp.Mirror(mp.Y, phase=+1)]




# In[16]:


sim = mp.Simulation(cell_size=cell_size,
                    geometry=geometry,
                    sources=sources,
                    k_point=k_point,
                    boundary_layers=pml_layers,
                    symmetries=symmetries,
                    resolution=resolution)
flux_object = sim.add_flux(freq, df, nfreq, fr)


sim.run(until=200)


combined_initial_flux = mp.get_fluxes(flux_object)[0]



# In[4]:


geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,height_SiO2), 
                    center=mp.Vector3(0,0,center_SiO2),
                    material=mp.Medium(index=n_SiO2)))

geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,height_Si_add), 
                    center=mp.Vector3(0,0,center_Si_add),
                    material=mp.Medium(index=n_Si)))


## initialize matrix for data collection ##
##########################################

num_holes = 15  # this is the number of pillars we will build
num_height = 15
min_r = 0.1
max_r = 0.245
min_h = 1.02
max_h = 5
data = np.zeros((4,num_holes))






# In[5]:


sim.reset_meep()

radius_data = {}
height_list = []

def run_sim(radius,height,sim) :
    print(radius)
    
    geometry.append(mp.Cylinder(radius=radius,
                        height=height,
                        axis=mp.Vector3(0,0,1),
                        center=mp.Vector3(0,0,center_hole),
                        material=mp.Medium(index=n_air)))
                
                
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        k_point=k_point,
                        boundary_layers=pml_layers,
                        symmetries=symmetries,
                        resolution=resolution)
                    #save_geometry(sim, radius, height)
                        
    flux_object = sim.add_flux(freq, df, nfreq, fr)  
    sim.run(until=200)
    res = sim.get_eigenmode_coefficients(flux_object, [1], eig_parity=mp.ODD_Y)
    coeffs = res.alpha

    flux = abs(coeffs[0,0,0]**2)
    phase = np.angle(coeffs[0,0,0]) 

    if phase<-np.pi:
        phase = phase +2*np.pi
    elif phase>-np.pi and phase<np.pi:
        phase = phase
    else:
        phase = phase - 2*np.pi
                                                    
   #radii_list.append(radius)
   # flux_list.append(flux)
   # phase_list.append(phase)
   # new_heightlist.append(height)
                                                                        
    
    sim.reset_meep()
    geometry.pop(-1)

    return phase, flux

def save_geometry(sim, radius, height):
    fig = plt.figure(figsize=(10,10))
    sim.plot2D(output_plane=plot_plane)

pbar = tqdm(total=num_height*num_holes,leave=False)
new_radiuslist =[]
for r, radius in enumerate(np.linspace(min_r,max_r,num_holes)):
    depth_list=[] 
    flux_list, phase_list=[],[]
    new_radiuslist.append(radius)
    for h, height in enumerate(np.linspace(min_h,max_h,num_height)):
        depth_list.append(height)
        phase, flux=run_sim(radius,height,sim)
        phase_list.append(phase)
        flux_list.append(flux)
        
        print(h,r)
        print("#########################################")
        print(f"phase: {phase} flux: {flux} radis:{radius} depth:{height}")
        pbar.update(1)
    radius_data[r] = {
            'radius' : radius,
            'flux' : flux_list,
            'phase' : phase_list,
            'depth' : depth_list,
            'new_radius' : new_radiuslist 
             }
       
pbar.close()

## pickle operations

for k,j in enumerate(np.linspace(0,num_holes-1,num_holes)):
    for v,l in enumerate(np.linspace(0,num_height-1, num_height)):
        x=radius_data[k]['depth']
        y=radius_data[k]['new_radius']
        z=radius_data[k]['phase']
        z_1=radius_data[k]['flux']

results_path = '/develop/results'
file_name = 'radius_data.pkl'
name = os.path.join(results_path,file_name)
pickle.dump(radius_data, open(name, "wb"))

