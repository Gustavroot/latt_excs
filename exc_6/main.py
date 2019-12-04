from simulation import Simulation
import numpy as np
import time

sim_params = dict()
# either 'ones' or 'random'
sim_params['gauge_links_init'] = 'random'
# using 1+1 for this simulation
sim_params['spacetime_dim'] = 2
# T,X,Y,...
sim_params['latt_size_per_dim'] = [12,12]
sim_params['rnd_seed'] = 38297

sim = Simulation(sim_params)

# parameters of the run
run_params = dict()
run_params['beta'] = 1.8
run_params['mc_steps'] = 2003000
run_params['skip'] = 50
run_params['burn_in'] = 3000

# Generation of the Markov Chain

start = time.time()
# run a Metropolis
sim.run(run_params)
end = time.time()
elapsed_time = end-start
print("")
print("Time for generation of the Markov Chain: "+str(elapsed_time))

# Post-processing

start = time.time()
# compute <P_avg>
plaq_stats = sim.compute_avg_plaq()
end = time.time()
elapsed_time = end-start
print("")
print("Time for computing <P_{avg}>: "+str(elapsed_time))
print("")
print("Results for <P_{avg}>:")
print("\t mean: "+str(plaq_stats[0]))
print("\t standard deviation: "+str(plaq_stats[1]))
print("")
