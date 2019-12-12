from simulation import Simulation
import numpy as np
import time


# Generation of the Markov Chain for the cases of interest

print("\nMARKOV CHAIN GENERATION:\n")

sims = list()

dims_to_sim = [(12,12), (32,32)]
betas_to_sim = [1.8, 12.8]
run_params_all = []

for i in range(len(dims_to_sim)):

    sim_params = dict()
    # either 'ones' or 'random'
    sim_params['gauge_links_init'] = 'random'
    # using 1+1 for this simulation
    sim_params['spacetime_dim'] = 2
    # T,X,Y,...
    sim_params['latt_size_per_dim'] = dims_to_sim[i]
    sim_params['rnd_seed'] = 3827 + i*62

    #sim = Simulation(sim_params)
    #sims.append(sim)
    sims.append( Simulation(sim_params) )

    # parameters of the run
    run_params = dict()
    run_params['beta'] = betas_to_sim[i]
    run_params['burn_in'] = 3000
    run_params['mc_steps'] = 500000 + run_params['burn_in']
    run_params['skip'] = 50

    run_params_all.append(run_params.copy())

    start = time.time()
    # run a Metropolis
    sims[i].run(run_params)
    end = time.time()
    elapsed_time = end-start
    print("Time for generation of the Markov Chain (dims = " + str(dims_to_sim[i]) + "): "+str(elapsed_time))


# Post-processing

print("\nPOST-PROCESSING:")

for i in range(len(sims)):

    # Make all links have their angle from 0 to 2*pi
    #sims[i].wrap_angle()

    print("\nTopological charge for latt = "+str(sims[i].latt_size_per_dim)+"...")

    start = time.time()
    Q = sims[i].compute_topological_charge()
    end = time.time()
    elapsed_time = end-start
    print("Time for computing Q: "+str(elapsed_time))
    sims[i].hist_plot(Q, "Q", run_params_all[i])
    sims[i].simple_plot(Q, "Q", run_params_all[i])
    print("...done.")

    print("\nAction for latt = "+str(sims[i].latt_size_per_dim)+"...")
    start = time.time()
    action = sims[i].compute_action(run_params_all[i]['beta'])
    end = time.time()
    elapsed_time = end-start
    print("Time for computing the action: "+str(elapsed_time))
    #print(action)
    sims[i].hist_plot(action, "action", run_params_all[i])
    sims[i].simple_plot(action, "action", run_params_all[i])
    print("...done.")
