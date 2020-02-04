from simulation import Simulation
from cmath import exp as cmplx_exp
from scipy.sparse.linalg import eigs as eig_sparse
from scipy.linalg import eig as eig_nonsparse
import matplotlib.pyplot as plt
import numpy as np
import time



# Define the set of values of Q
#Q_values = [float(i) for i in range(-6,6,1)]
Q_values = [-6.0,-3.0,-1.0,0.0,1.0,3.0,6.0]

# There are four cases to be studied:
#	- naive, no gauge interactions
#	- Wilson, no gauge interactions
#	- naive, with gauge interactions
#	- Wilson, with gauge interactions

dims = (20,20)
#dims = (4,4)

# NAIVE, NO GAUGE INTERACTIONS

print("")

for i in range(1):

    print("Naive with no gauge interactions...")

    # 0. create simulation object

    sim_params = dict()
    # either 'ones' or 'random'
    sim_params['gauge_links_init'] = 'ones'
    # using 1+1 for this simulation
    sim_params['spacetime_dim'] = 2
    # T,X,Y,...
    sim_params['latt_size_per_dim'] = dims
    # creating the simulation object
    sim = Simulation(sim_params)

    # 1. create a gauge config with top charge = Q
    #sim.build_U_with_Q(3.0)
    sim.MARKOV_CHAIN = [sim.gauge_links]
    #top_charge = sim.compute_topological_charge()
    # exponentiating the gauge links' phases
    sim.gauge_links_exp = np.copy(sim.gauge_links)
    sim.gauge_links_exp = sim.gauge_links_exp.flatten()
    sim.gauge_links_exp = np.array([cmplx_exp(complex(0.0,a)) for a in sim.gauge_links_exp])
    sim.gauge_links_exp = sim.gauge_links_exp.reshape([sim.nr_latt_sites,2])

    # 2. construct M

    # after this following line, sim.dirac_matrix is a sparse construction of D
    # also, we use m=0.0
    sim.buildM(sim.gauge_links, 0.0, 0.0)
    sim.dirac_matrix = sim.dirac_matrix*(-1.0)

    # 3. call eigensolver
    evals_all,evecs_all = eig_nonsparse(sim.dirac_matrix.toarray())

    # 4. plot spectrum of M
    X = [x.real for x in evals_all]
    Y = [y.imag for y in evals_all]
    plt.scatter(X,Y,color='red')
    plt.savefig("naive_nongauge.png")
    plt.clf()

    print("...done")

# WILSON, NO GAUGE INTERACTIONS

print("")

for i in range(1):

    print("Wilson with no gauge interactions...")

    # 0. create simulation object

    sim_params = dict()
    # either 'ones' or 'random'
    sim_params['gauge_links_init'] = 'ones'
    # using 1+1 for this simulation
    sim_params['spacetime_dim'] = 2
    # T,X,Y,...
    sim_params['latt_size_per_dim'] = dims
    # creating the simulation object
    sim = Simulation(sim_params)

    # 1. create a gauge config with top charge = Q
    #sim.build_U_with_Q(3.0)
    sim.MARKOV_CHAIN = [sim.gauge_links]
    #top_charge = sim.compute_topological_charge()
    # exponentiating the gauge links' phases
    sim.gauge_links_exp = np.copy(sim.gauge_links)
    sim.gauge_links_exp = sim.gauge_links_exp.flatten()
    sim.gauge_links_exp = np.array([cmplx_exp(complex(0.0,a)) for a in sim.gauge_links_exp])
    sim.gauge_links_exp = sim.gauge_links_exp.reshape([sim.nr_latt_sites,2])

    # 2. construct M

    # after this following line, sim.dirac_matrix is a sparse construction of D
    # also, we use m=0.0
    sim.buildM(sim.gauge_links, 1.0, 0.0)
    sim.dirac_matrix = sim.dirac_matrix*(-1.0)

    # 3. call eigensolver
    evals_all,evecs_all = eig_nonsparse(sim.dirac_matrix.toarray())

    # 4. plot spectrum of M
    X = [x.real for x in evals_all]
    Y = [y.imag for y in evals_all]
    plt.scatter(X,Y,color='red')
    #plt.xlim(-1.0e-14,1.0e-14)
    #plt.ylim(-1.0e-14,1.0e-14)
    plt.savefig("Wilson_nongauge.png")
    plt.clf()

    print("...done")

# NAIVE, WITH GAUGE INTERACTIONS

print("")

# Loop over all desired values of Q
for Q in Q_values:

    print("Naive with gauge interactions (Q="+str(Q)+")...")

    # 0. create simulation object

    sim_params = dict()
    # either 'ones' or 'random'
    sim_params['gauge_links_init'] = 'ones'
    # using 1+1 for this simulation
    sim_params['spacetime_dim'] = 2
    # T,X,Y,...
    sim_params['latt_size_per_dim'] = dims
    # creating the simulation object
    sim = Simulation(sim_params)

    # 1. create a gauge config with top charge = Q
    sim.build_U_with_Q(Q)
    sim.MARKOV_CHAIN = [sim.gauge_links]
    top_charge = sim.compute_topological_charge()
    # exponentiating the gauge links' phases
    sim.gauge_links_exp = np.copy(sim.gauge_links)
    sim.gauge_links_exp = sim.gauge_links_exp.flatten()
    sim.gauge_links_exp = np.array([cmplx_exp(complex(0.0,a)) for a in sim.gauge_links_exp])
    sim.gauge_links_exp = sim.gauge_links_exp.reshape([sim.nr_latt_sites,2])

    # 2. construct M

    # after this following line, sim.dirac_matrix is a sparse construction of D
    # also, we use m=0.0
    sim.buildM(sim.gauge_links, 0.0, 0.0)
    sim.dirac_matrix = sim.dirac_matrix*(-1.0)

    # 3. call eigensolver
    evals_all,evecs_all = eig_nonsparse(sim.dirac_matrix.toarray())

    # 4. plot spectrum of M
    X = [x.real for x in evals_all]
    Y = [y.imag for y in evals_all]
    plt.scatter(X,Y,color='red')
    plt.savefig("naive_gauge_Q"+str(round(top_charge[0]))+".png")
    plt.clf()

    print("...done")

# WILSON, WITH GAUGE INTERACTIONS

print("")

# Loop over all desired values of Q
for Q in Q_values:

    print("Wilson with gauge interactions (Q="+str(Q)+")...")

    # 0. create simulation object

    sim_params = dict()
    # either 'ones' or 'random'
    sim_params['gauge_links_init'] = 'ones'
    # using 1+1 for this simulation
    sim_params['spacetime_dim'] = 2
    # T,X,Y,...
    sim_params['latt_size_per_dim'] = dims
    # creating the simulation object
    sim = Simulation(sim_params)

    # 1. create a gauge config with top charge = Q
    sim.build_U_with_Q(Q)
    sim.MARKOV_CHAIN = [sim.gauge_links]
    top_charge = sim.compute_topological_charge()
    # exponentiating the gauge links' phases
    sim.gauge_links_exp = np.copy(sim.gauge_links)
    sim.gauge_links_exp = sim.gauge_links_exp.flatten()
    sim.gauge_links_exp = np.array([cmplx_exp(complex(0.0,a)) for a in sim.gauge_links_exp])
    sim.gauge_links_exp = sim.gauge_links_exp.reshape([sim.nr_latt_sites,2])

    # 2. construct M

    # after this following line, sim.dirac_matrix is a sparse construction of D
    # also, we use m=0.0
    sim.buildM(sim.gauge_links, 1.0, 0.0)
    sim.dirac_matrix = sim.dirac_matrix*(-1.0)

    # 3. call eigensolver
    evals_all,evecs_all = eig_nonsparse(sim.dirac_matrix.toarray())

    # 4. plot spectrum of M
    X = [x.real for x in evals_all]
    Y = [y.imag for y in evals_all]
    plt.scatter(X,Y,color='red')
    plt.savefig("Wilson_gauge_Q"+str(round(top_charge[0]))+".png")
    plt.clf()
    
    print("...done")
