# Check for dependencies
from simulation import check_if_pack_installed

packages_needed = ['matplotlib', 'scipy']
for pack in packages_needed:
    check_if_pack_installed(pack)

from simulation import Simulation

# Simulation params
params = dict()
params['N'] = 50
params['w'] = 0.2
params['burn_in'] = 2500
params['rnd_seed'] = 10378

# Create and init simulation
sim = Simulation(params)
sim.set_params(params)

run_setup = dict()
run_setup['mc_steps'] = 20000
run_setup['prop_dist'] = 'gaussian'

# RUN #1

# Use of prior distribution
sim.init_use_prior_pdf(run_setup['prop_dist'])
# Run the simulation a certain number of steps
sim.run(run_setup['mc_steps'], run_setup['prop_dist'])
# Plot a certain variable over MC time
sim.plot('S_E', 'sim_action1.png', run_setup)
sim.plot('wave_function', 'sim_wave_function.png', run_setup)

# Clean the simulation to run a new one
sim.clean()
#params['burn_in'] = 1500
sim.set_params(params)

# RUN #2

# Use of prior distribution
sim.init_use_prior_pdf(run_setup['prop_dist'])
# Run the simulation a certain number of steps
sim.run(run_setup['mc_steps'], run_setup['prop_dist'])
# Plot a certain variable over MC time
sim.plot('S_E', 'sim_action2.png', run_setup)
sim.plot('wave_function', 'sim_wave_function.png', run_setup)
