from simulation import Simulation


sim_params = dict()
sim_params['n_s'] = 8
sim_params['n_t'] = 8
sim_params['m'] = 0.4
sim_params['rnd_seed'] = 4568

sim = Simulation(sim_params)

run_params = dict()
run_params['burn_in'] = 3000
run_params['mc_steps'] = 10003000
run_params['skip_length'] = 100

# Generate Markov Chain configurations by using Metropolis
sim.run(run_params)

# post-processing
sim.apply_ft()
sim.compute_corrs()
sim.compute_eff_energies()

# displaying results
sim.plot("corrs", "corrs.png", run_params)
sim.plot("eff_energies", "eff_energies.png", run_params)
