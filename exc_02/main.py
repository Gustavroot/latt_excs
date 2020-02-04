# Check for dependencies
import time
from simulation import check_if_pack_installed

packages_needed = ['matplotlib', 'scipy']
for pack in packages_needed:
    check_if_pack_installed(pack)

from simulation import Simulation

# Simulation params
params = dict()
params['N'] = 20
params['w'] = 0.5
params['burn_in'] = 2500
params['rnd_seed'] = 10378
params['nr_bootstrap'] = 200

# Create and init simulation
sim = Simulation()
sim.set_params(params)

run_setup = dict()
run_setup['mc_steps'] = 82500
run_setup['prior_dist'] = 'gaussian'
run_setup['prop_dist'] = 'gaussian_all'
run_setup['delta'] = 100

# Use of prior distribution
sim.init_use_prior_pdf(run_setup['prior_dist'])
# Run the simulation a certain number of steps
start = time.time()
sim.run(run_setup['mc_steps'], run_setup['prop_dist'], run_setup['delta'])
end = time.time()
print("Time taken for constructing the Markov Chain simulation: "+str(end-start))

sim.compute_corrs()
sim.eff_energy_exp_all()

sim.plot('eff_energy_exp', 'sim_eff_energ.png', run_setup)
sim.plot('corrs', 'corr_function.png', run_setup)

# Plot a certain variable over MC time
#sim.plot('S_E', 'sim_action1.png', run_setup)
#sim.plot('wave_function', 'sim_wave_function1.png', run_setup)

# Clean the simulation to run a new one
#sim.clean()
#params['burn_in'] = 1500
#params['rnd_seed'] = 184
#sim.set_params(params)
