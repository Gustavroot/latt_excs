from simulation import Simulation

# Simulation params
params = dict()
params['N'] = 100
params['w'] = 0.2
#params['rnd_seed'] = 103

# Create and init simulation
sim = Simulation(params)

run_setup = dict()
run_setup['mc_steps'] = 10000
run_setup['prop_dist'] = 'gaussian'

# Use of prior distribution
sim.init_use_prior_pdf(run_setup['prop_dist'])

# Run the simulation a certain number of steps
sim.run(run_setup['mc_steps'], run_setup['prop_dist'])

# Plot a certain variable over MC time
sim.plot('S_E', 'sim.png', run_setup)
