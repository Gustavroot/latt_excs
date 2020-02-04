from simulation import Simulation
import numpy as np

sim_params = dict()
# either 'ones' or 'random'
sim_params['gauge_links_init'] = 'ones'
# using 1+1 for this simulation
sim_params['spacetime_dim'] = 2
# T,X,Y,...
sim_params['latt_size_per_dim'] = [4,4]
sim_params['rnd_seed'] = 4728

sim = Simulation(sim_params)

sim.set_gauge_links("random")

print("")
print("Action before gauge transf: "+str(sim.compute_gauge_action()))

# compute gauge transformation on all links
sim.full_gauge_transf()

print("")
print("Action after gauge transf: "+str(sim.compute_gauge_action()))
print("")


"""

# OLD TESTS

plaq_point = (0,1,6)
phase_buff = sim.compute_single_plaquette_phase(plaq_point[0],plaq_point[1],plaq_point[2])
print("Phase for "+str(plaq_point)+": "+str(phase_buff))
plaq_buff = sim.compute_plaquette_from_phase(phase_buff)
print("Plaquette value for "+str(plaq_point)+": "+str(plaq_buff))

# only 'random' allowed for the input param
sim.init_gauge_transf('random')

# compute gauge transformation on all links
for i in range(np.prod(sim.latt_size_per_dim)):
    for mu in range(sim.spacetime_dim):
        # compute the transformed phase
        changed_phase = sim.gauge_transf_single_link(mu,i)
        # change the corresponding link
        sim.gauge_links[i,mu] = changed_phase

# check if plaquette hasn't changed
print("Changed phase = "+str(changed_phase))
plaq_point = (0,1,6)
phase_buff = sim.compute_single_plaquette_phase(plaq_point[0],plaq_point[1],plaq_point[2])
changed_plaq = sim.compute_plaquette_from_phase(phase_buff)
print("Changed plaq: "+str(changed_plaq))

"""
