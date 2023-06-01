import os
import pickle
import shutil 

import meep as mp
import numpy as np

from tqdm import tqdm
import math
from IPython import embed

resolution = 20

n_air = 1.0000003
n_Si = 3.48
n_SiO2 = 1.45

a = 0.5   # lattice period 

pml_thickness = 0.780
Si = 5       # eventually we're going to vary this value so the following values will change dynamically
SiO2 = 0.6 * Si         # so the height of Si02 (high index material embedded in Si) is always 25percent of Si
air_buffer = 0.4 * SiO2          # and the air buffer is 40 percent of SiO2
air_block = pml_thickness*2 + Si + air_buffer*2   # this is the length of the material plus pml and a 0.5 micron air buffer above and below the material
air_hole = air_block

cell_x = a
cell_y = a
cell_z = air_block

center_Si = 0
center_air = 0
center_SiO2 = 0.5*SiO2

k_point = mp.Vector3(0,0,0)

wavelength = 1.55
freq = 1 / wavelength

##### build geometry ########
#############################

geometry=[mp.Block(size=mp.Vector3(mp.inf,mp.inf,air_block),
                    center=mp.Vector3(0,0,center_air),
                    material=mp.Medium(index=n_air))]

## initialize matrix for data collection ##
##########################################

num_holes = 15 # this is the number of holes we will build
data = np.zeros((3,num_holes))

## set up and build source ##
############################

center_source = -0.5*Si - 0.4*air_buffer
source_cmpt = mp.Ey

sources = [mp.Source(mp.ContinuousSource(frequency=freq),
                    component=source_cmpt,
                    center=mp.Vector3(0,0,center_source),
                    size=mp.Vector3(cell_x,cell_y,0))]

## Set up simulation ##
######################

cell_size = mp.Vector3(cell_x,cell_y,cell_z)
pml_layers = [mp.PML(thickness = pml_thickness, direction = mp.Z)]

if source_cmpt == mp.Ey:
    symmetries = [mp.Mirror(mp.X, phase=+1), #epsilon has mirror symmetry in x and y, phase doesn't matter
                  mp.Mirror(mp.Y, phase=-1)] #but sources have -1 phase when reflected normal to their direction
elif src_cmpt == mp.Ex:                      #use of symmetries important here, significantly speeds up sim
    symmetries = [mp.Mirror(mp.X, phase=-1),
                  mp.Mirror(mp.Y, phase=+1)]
elif src_cmpt == mp.Ez:
    symmetries = [mp.Mirror(mp.X, phase=+1),
                  mp.Mirror(mp.Y, phase=+1)]

sim = mp.Simulation(cell_size=cell_size,
                    geometry=geometry,
                    sources=sources,
                    k_point=k_point,
                    boundary_layers=pml_layers,
                    symmetries=symmetries,
                    resolution=resolution)

## set up and build flux monitor ##
####################################

nfreq = 1
df = 0
fr_center = 0.5*Si + 0.4*air_buffer
fr = mp.FluxRegion(center=mp.Vector3(0,0,fr_center), 
            size=mp.Vector3(cell_x, cell_y, 0))

flux_object = sim.add_flux(freq, df, nfreq, fr)

sim.run(until=200)

initial_flux = mp.get_fluxes(flux_object)[0]     # initial flux

sim.reset_meep()

geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,Si), 
                   center=mp.Vector3(0,0,center_Si),
                   material=mp.Medium(index=n_Si)))

geometry.append(mp.Block(size=mp.Vector3(mp.inf,mp.inf,SiO2),
                    center=mp.Vector3(0,0,center_SiO2),
                    material=mp.Medium(index=n_SiO2)))

pbar = tqdm(total=num_holes,leave=False)
min = 0.0
max = 0.240
for i,radius in enumerate(np.linspace(min,max,num=num_holes)):
    geometry.append(mp.Cylinder(radius=radius,
                        height=cell_z,
                        axis=mp.Vector3(0,0,1),
                        center=mp.Vector3(0,0,0),
                        material=mp.Medium(index=n_air)))

    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        sources=sources,
                        k_point=k_point,
                        boundary_layers=pml_layers,
                        symmetries=symmetries,
                        resolution=resolution)
    
    flux_object = sim.add_flux(freq, df, nfreq, fr)  

    sim.run(until=200)
    
    res = sim.get_eigenmode_coefficients(flux_object, [1], eig_parity=mp.ODD_Y)
    coeffs = res.alpha

    flux = abs(coeffs[0,0,0]**2)
    phase = np.angle(coeffs[0,0,0]) 
    
    data[0,i] = radius
    data[1,i] = flux
    data[2,i] = phase
    
    if(radius!=max):
        sim.reset_meep()
        print(f"i= {i},radius={radius}")
        geometry.pop(-1)
    pbar.update(1)
pbar.close()

radii = data[0,:]
flux_list = data[1,0:] / initial_flux
phase_list = data[2,0:]

pickle.dump(data, open("data.pkl","wb"))
pickle.dump(initial_flux, open("initial_flux.pkl","wb"))
