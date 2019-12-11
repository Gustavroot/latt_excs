from simulation import Simulation
import numpy as np
import time


# Generation of the Markov Chain for the cases of interest

print("\nMARKOV CHAIN GENERATION:\n")

sims = list()

dims_to_sim = [(12,12), (32,32)]
betas_to_sim = [1.8, 2.8]
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
    run_params['mc_steps'] = 2000 + run_params['burn_in']
    run_params['skip'] = 50

    run_params_all.append(run_params.copy())

    start = time.time()
    # run a Metropolis
    sims[i].run(run_params)
    end = time.time()
    elapsed_time = end-start
    print("Time for generation of the Markov Chain (dims = " + str(dims_to_sim[i]) + "): "+str(elapsed_time))


# Post-processing

print("\nPOST-PROCESSING:\n")

for i in range(len(sims)):

    print("\nTopological charge for latt = "+str(sims[i].latt_size_per_dim)+"...")
    Q = sims[i].compute_topological_charge()
    #print(Q)
    sims[i].hist_plot(Q, "Q", run_params_all[i])
    sims[i].simple_plot(Q, "Q", run_params_all[i])
    print("...done.")

    print("\nAction for latt = "+str(sims[i].latt_size_per_dim)+"...")
    action = sims[i].compute_action()
    #print(action)
    sims[i].hist_plot(action, "action", run_params_all[i])
    sims[i].simple_plot(action, "action", run_params_all[i])
    print("...done.")
