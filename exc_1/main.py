# Check for dependencies
import time
from simulation import check_if_pack_installed

packages_needed = ['matplotlib', 'scipy']
for pack in packages_needed:
    check_if_pack_installed(pack)

from simulation import Simulation

# Simulation params
params = dict()
params['N'] = 100
params['w'] = 0.2
params['burn_in'] = 2500
params['rnd_seed'] = 10378

# Create and init simulation
sim = Simulation()
sim.set_params(params)

run_setup = dict()
run_setup['mc_steps'] = 60000
run_setup['prior_dist'] = 'gaussian'
run_setup['prop_dist'] = 'gaussian_all'

# RUN #1

# Use of prior distribution
sim.init_use_prior_pdf(run_setup['prior_dist'])
# Run the simulation a certain number of steps
start = time.time()
sim.run(run_setup['mc_steps'], run_setup['prop_dist'])
end = time.time()
print("Time taken for first simulation: "+str(end-start))
# Plot a certain variable over MC time
sim.plot('S_E', 'sim_action1.png', run_setup)
sim.plot('wave_function', 'sim_wave_function1.png', run_setup)

# Clean the simulation to run a new one
sim.clean()
#params['burn_in'] = 1500
params['rnd_seed'] = 184
sim.set_params(params)

# RUN #2

run_setup['prop_dist'] = 'gaussian_one'

# Use of prior distribution
sim.init_use_prior_pdf(run_setup['prior_dist'])
# Run the simulation a certain number of steps
start = time.time()
sim.run(run_setup['mc_steps'], run_setup['prop_dist'])
end = time.time()
print("Time taken for second simulation: "+str(end-start))
# Plot a certain variable over MC time
sim.plot('S_E', 'sim_action2.png', run_setup)
sim.plot('wave_function', 'sim_wave_function2.png', run_setup)
