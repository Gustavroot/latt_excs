import numpy as np

from cmath import exp as cmplx_exp
from cmath import log as cmplx_log
from math import cos, exp

from scipy import sparse

import matplotlib
import matplotlib.pyplot as plt

import sys


# TODO:

#	1. double-check generalities of the code for correctness
#	2. complete step c of assignment (this tests for correctness again)




"""

Structure of spacetime:

t ---> self.latt_size_per_dim[0]
x ---> self.latt_size_per_dim[1]

t
^
|
|
|
|
|
| 8 9 ...
| 0 1 2 3 4 5 6 7
--------------------> x

"""



class Simulation:

    gauge_links = []
    gauge_transf = []
    
    """
    the use of self.neighbors_table is necessary. Take the following timings:

    For the case of a lattice of size 1024x1024:

     ** without neighbors_table: 39.3198750019 seconds
     ** with neighbors_table: 13.6272730827 seconds
    """
    neighbors_table_forward = []
    neighbors_table_backward = []
    neighbors_table_crossed = []

    MARKOV_CHAIN = []


    def __init__(self, params):

        self.spacetime_dim = params['spacetime_dim']

        if len(params['latt_size_per_dim']) != self.spacetime_dim:
            raise Exception("Check your input parameter 'latt_size_per_dim'.")
        else:
            self.latt_size_per_dim = np.array(params['latt_size_per_dim'])

        self.nr_latt_sites = np.prod(self.latt_size_per_dim)

        # create empty NumPy array (virtual memory advantages kick-in)
        self.gauge_links = np.empty([self.nr_latt_sites,self.spacetime_dim])
        self.gauge_transf = np.empty([self.nr_latt_sites,1])

        self.neighbors_table_forward = np.empty([self.nr_latt_sites,self.spacetime_dim], dtype='int')
        self.neighbors_table_backward = np.empty([self.nr_latt_sites,self.spacetime_dim], dtype='int')
        # FIXME: hardcoded '2', due to nu!=mu and spacetime_dim=2
        self.neighbors_table_crossed = np.empty([self.nr_latt_sites,2], dtype='int')

        if 'rnd_seed' in params:
            np.random.seed(params['rnd_seed'])

        # initialize gauge links
        self.set_gauge_links(params['gauge_links_init'])
        # only 'random' allowed for the input param
        self.init_gauge_transf('random')

        # pre-build table with indices of neighbors
        self.build_neighbors_table_forward()
        self.build_neighbors_table_backward()
        self.build_neighbors_table_crossed()

        self.gauge_links_exp = []

        self.gamma0 = [[1.0,0.0],[0.0,-1.0]]
        self.gamma0 = np.array(self.gamma0)
        self.gamma1 = [[0.0,complex(0.0,-1.0)],[complex(0.0,1.0),0.0]]
        self.gamma1 = np.array(self.gamma1)
        self.gammaIdent = [[1.0,0.0],[0.0,1.0]]
        self.gammaIdent = np.array(self.gammaIdent)

        self.dirac_matrix = []


    # <init_mode> \in {'ones','random'}
    def set_gauge_links(self, init_mode):

        if init_mode == 'ones':
            # for gauge links equal to 'ones', their phases must be 'zeroes'
            self.gauge_links.fill(0.0)
        elif init_mode == 'random':
            self.gauge_links = np.random.uniform(0,2.0*np.pi,self.gauge_links.shape)
        else:
            raise Exception("Chosen init mode for gauge links not available.")


    # pre-build a 'forward' neighbors table to reduce function-calling overhead
    def build_neighbors_table_forward(self):

        for idx in range(self.nr_latt_sites):
            for mu in range(self.spacetime_dim):
                # convert idx into lattice coordinates
                coords_n = self.idx_to_coords(idx)
                # get index for n+mu
                coords_n_p_mu = np.copy(coords_n)
                coords_n_p_mu[mu] += 1
                # boundary conditions
                coords_n_p_mu[mu] = coords_n_p_mu[mu]%self.latt_size_per_dim[mu]
                idx_n_p_mu = self.coords_to_idx(coords_n_p_mu)
                self.neighbors_table_forward[idx][mu] = idx_n_p_mu


    def build_neighbors_table_backward(self):

        for idx in range(self.nr_latt_sites):
            for mu in range(self.spacetime_dim):
                # convert idx into lattice coordinates
                coords_n = self.idx_to_coords(idx)
                # get index for n+mu
                coords_n_m_mu = np.copy(coords_n)

                #coords_n_p_mu[mu] -= 1
                # boundary conditions
                #coords_n_p_mu[mu] = coords_n_p_mu[mu]%self.latt_size_per_dim[mu]

                # -1 in mu direction
                coords_n_m_mu[mu] = (self.latt_size_per_dim[mu]+coords_n_m_mu[mu]-1)%self.latt_size_per_dim[mu]

                idx_n_m_mu = self.coords_to_idx(coords_n_m_mu)
                self.neighbors_table_backward[idx][mu] = idx_n_m_mu


    def build_neighbors_table_crossed(self):

        for idx in range(self.nr_latt_sites):

            for mu in range(2):

                nu = int( not mu )

                # 1st case
                #mu=0
                #nu=1
                # 2nd case
                #mu=1
                #nu=0

                # convert idx into lattice coordinates
                coords_n = self.idx_to_coords(idx)
                # get index for n
                coords_n_m_nu_p_mu = np.copy(coords_n)

                # neighbor in position n-nu+mu
                coords_n_m_nu_p_mu[nu] = (self.latt_size_per_dim[nu]+coords_n_m_nu_p_mu[nu]-1)%self.latt_size_per_dim[nu]
                coords_n_m_nu_p_mu[mu] += 1
                coords_n_m_nu_p_mu[mu] = coords_n_m_nu_p_mu[mu]%self.latt_size_per_dim[mu]

                idx_n_m_nu_p_mu = self.coords_to_idx(coords_n_m_nu_p_mu)
                self.neighbors_table_crossed[idx][mu] = idx_n_m_nu_p_mu


    def shuffle_gauge_links(self):
        self.gauge_links = np.random.uniform(0,2.0*np.pi,self.gauge_links.shape)


    # <idx> is the index of the lattice site
    def compute_single_plaquette_phase(self, mu, nu, idx):

        if mu>=self.spacetime_dim or nu>=self.spacetime_dim:
            raise Exception("Values of mu and nu in self.compute_single_plaquette(...) not allowed.")

        """
        # convert idx into lattice coordinates
        coords_n = self.idx_to_coords(idx)
        # get index for n+mu
        coords_n_p_mu = np.copy(coords_n)
        coords_n_p_mu[mu] += 1
        # boundary conditions
        coords_n_p_mu[mu] = coords_n_p_mu[mu]%self.latt_size_per_dim[mu]
        idx_n_p_mu = self.coords_to_idx(coords_n_p_mu)
        # get index for n+nu
        coords_n_p_nu = np.copy(coords_n)
        coords_n_p_nu[nu] += 1
        # boundary conditions
        coords_n_p_nu[nu] = coords_n_p_nu[nu]%self.latt_size_per_dim[nu]
        idx_n_p_nu = self.coords_to_idx(coords_n_p_nu)
        """

        plaq_value = 0.0
        # non-daggered entries
        plaq_value += self.gauge_links[idx][mu]
        idx_n_p_mu = self.neighbors_table_forward[idx][mu]
        plaq_value += self.gauge_links[idx_n_p_mu][nu]
        # daggered entries
        idx_n_p_nu = self.neighbors_table_forward[idx][nu]
        plaq_value -= self.gauge_links[idx_n_p_nu][mu]
        plaq_value -= self.gauge_links[idx][nu]

        return plaq_value


    # <idx> is the index of the lattice site
    def compute_single_plaquette_phase_gen(self, mu, nu, idx, gauge_links):

        if mu>=self.spacetime_dim or nu>=self.spacetime_dim:
            raise Exception("Values of mu and nu in self.compute_single_plaquette(...) not allowed.")

        plaq_value = 0.0
        # non-daggered entries
        plaq_value += gauge_links[idx][mu]
        idx_n_p_mu = self.neighbors_table_forward[idx][mu]
        plaq_value += gauge_links[idx_n_p_mu][nu]
        # daggered entries
        idx_n_p_nu = self.neighbors_table_forward[idx][nu]
        plaq_value -= gauge_links[idx_n_p_nu][mu]
        plaq_value -= gauge_links[idx][nu]

        return plaq_value


    # TODO: extend to dimensionality greater than 2
    def idx_to_coords(self, idx):

        t = idx/self.latt_size_per_dim[1]
        x = idx%self.latt_size_per_dim[1]

        return np.array([t,x])


    # TODO: extend to dimensionality greater than 2
    # <coords> comes in the form {T_value, X_value, Y_value, ...}
    def coords_to_idx(self, coords):

        idx = 0
        idx += coords[0]*self.latt_size_per_dim[1]
        idx += coords[1]

        return idx


    def compute_plaquette_from_phase(self, phase):
        return cmplx_exp(complex(0.0,phase))


    def init_gauge_transf(self, init_mode):
        if init_mode == 'random':
            self.gauge_transf = np.random.uniform(0,2.0*np.pi,self.gauge_transf.shape)
        else:
            raise Exception("Chosen init mode for gauge transformations not available.")


    def shuffle_gauge_transf(self):
        self.gauge_transf = np.random.uniform(0,2.0*np.pi,self.gauge_transf.shape)


    # TODO: move gauge transformation outside as input param of this method?
    def gauge_transf_single_link(self, mu, idx):

        """
        # convert idx into lattice coordinates
        coords_n = self.idx_to_coords(idx)
        # get index for n+mu
        coords_n_p_mu = np.copy(coords_n)
        coords_n_p_mu[mu] += 1
        # boundary conditions
        coords_n_p_mu[mu] = coords_n_p_mu[mu]%self.latt_size_per_dim[mu]
        idx_n_p_mu = self.coords_to_idx(coords_n_p_mu)
        """

        idx_n_p_mu = self.neighbors_table_forward[idx][mu]

        phase = 0.0
        phase += self.gauge_transf[idx][0]
        phase += self.gauge_links[idx][mu]
        # daggered term
        phase -= self.gauge_transf[idx_n_p_mu][0]

        return phase


    def wrap_angle(self):

        for gauge_links in self.MARKOV_CHAIN:

            for idx in range(self.nr_latt_sites):
                for mu in range(self.spacetime_dim):
                    gauge_links[idx][mu] = gauge_links[idx][mu]%(2*np.pi)


    # this method returns S^{gauge}_{E} / \beta, for a given Gauge config
    def compute_gauge_action(self, gauge_links):

        action = 0.0
        for idx in range(self.nr_latt_sites):
            for nu in range(self.spacetime_dim):
                for mu in range(nu):

                    buff_nr = complex(1.0,0.0) - self.compute_plaquette_from_phase( self.compute_single_plaquette_phase_gen(mu,nu,idx,gauge_links) )
                    buff_nr = buff_nr.real

                    # take real part directly
                    #buff_nr = 1.0 - cos( self.compute_single_plaquette_phase(mu,nu,idx) )
                    #buff_nr = 1.0 - cos( self.compute_single_plaquette_phase_gen(mu, nu, idx, gauge_links) )

                    action += buff_nr

        return action


    # This return an array with the action computed for ALL Gauge configs
    def compute_action(self, beta):

        action_values = []

        for i in range(len(self.MARKOV_CHAIN)):
            action_values.append( self.compute_gauge_action(self.MARKOV_CHAIN[i]) )

        action_values = np.array(action_values)
        action_values *= beta

        return action_values


    def full_gauge_transf(self):
        for i in range(self.nr_latt_sites):
            for mu in range(self.spacetime_dim):
                # compute the transformed phase
                changed_phase = self.gauge_transf_single_link(mu,i)
                # change the corresponding link
                self.gauge_links[i,mu] = changed_phase


    def compute_staple_dagg(self, idx, mu):

        # FIXME: the following line is hardcoded for the case dim=2
        nu = int(not mu)

        # daggered terms
        phase1 = -self.gauge_links[ self.neighbors_table_crossed[idx][mu] ][nu]
        #print(self.neighbors_table_backward[idx][nu])
        phase1 -= self.gauge_links[ self.neighbors_table_backward[idx][nu] ][mu]
        # non-daggered term
        phase1 += self.gauge_links[ self.neighbors_table_backward[idx][nu] ][nu]

        # non-daggered term
        phase2 = self.gauge_links[ self.neighbors_table_forward[idx][mu] ][nu]
        # daggered terms
        phase2 -= self.gauge_links[ self.neighbors_table_forward[idx][nu] ][mu]
        phase2 -= self.gauge_links[ idx ][nu]

        term1 = cmplx_exp(complex(0.0,phase1))
        term2 = cmplx_exp(complex(0.0,phase2))

        return term1+term2


    def compute_gauge_action_change(self, beta, phase_new, phase_old, staple_dagg):

        u_new = cmplx_exp(complex(0.0,phase_new))
        u_old = cmplx_exp(complex(0.0,phase_old))

        term = (u_new - u_old)*staple_dagg

        return -beta*term.real


    # implementation of a Metropolis on the Gauge part of the system
    def run(self, params):

        beta = params['beta']
        mc_steps = params['mc_steps']
        burn_in = params['burn_in']
        skip = params['skip']

        # some verbosity
        print("Parameters of the run:")
        print("\t beta: "+str(beta))
        print("\t mc_steps: "+str(mc_steps))
        print("\t burn_in: "+str(burn_in))
        print("\t skip: "+str(skip))
        print("\t effective number of 'independent' configurations: "+str((mc_steps-burn_in)/skip))

        if burn_in>mc_steps:
            raise Exception("ERROR: burn_in can't be greater than mc_steps.")

        # burn-in
        print("Burn-in...")
        for i in range(burn_in):
            idx = i % self.nr_latt_sites
            for mu in range(self.spacetime_dim):
                # 1. proposal step
                changed_phase = np.random.normal(self.gauge_links[idx][mu], 0.1, 1)[0]
                # 2. compute change in action
                staple_dagg = self.compute_staple_dagg(idx, mu)
                action_change = self.compute_gauge_action_change( beta, changed_phase, self.gauge_links[idx][mu], staple_dagg )
                # 3. compute acceptance probability
                accept_prob = exp( -action_change )
                # 4. accept or reject
                u = np.random.uniform(size=1)[0]
                if u <= accept_prob:
                    # accept
                    self.gauge_links[idx][mu] = changed_phase
        print("...done.")

        self.MARKOV_CHAIN = []

        w = 0
        progr_prev = 0
        # after thermalization:
        for i in range(mc_steps-burn_in):
            idx = i % self.nr_latt_sites
            for mu in range(self.spacetime_dim):

                # 1. proposal step
                changed_phase = np.random.normal(self.gauge_links[idx][mu], 0.1, 1)[0]

                # 2. compute change in action
                staple_dagg = self.compute_staple_dagg(idx, mu)
                action_change = self.compute_gauge_action_change( beta, changed_phase, self.gauge_links[idx][mu], staple_dagg )

                # 3. compute acceptance probability
                accept_prob = exp( -action_change )

                # 4. accept or reject
                u = np.random.uniform(size=1)[0]

                # every skip-steps
                if i%skip == 0:
                    self.MARKOV_CHAIN.append(np.copy(self.gauge_links))
                    self.gauge_links = self.MARKOV_CHAIN[len(self.MARKOV_CHAIN)-1]

                if u <= accept_prob:
                    self.gauge_links[idx][mu] = changed_phase
                #else:
                #    pass

            # print progress
            progr = int((float(w) / float(mc_steps-burn_in))*100.0)
            if progr>0 and progr!=progr_prev and progr%2==0:
                print(str(progr)+"% ..."),
                sys.stdout.flush()
                progr_prev = progr
            w += 1
        print("")


    def compute_topological_charge(self):

        Q_values = []

        mu=1
        nu=0
        for i in range(len(self.MARKOV_CHAIN)):

            gauge_links = self.MARKOV_CHAIN[i]

            rel_sum = complex(0.0,0.0)
            #rel_sum = 0.0
            for idx in range(self.nr_latt_sites):

                phase_buff = self.compute_single_plaquette_phase_gen(mu, nu, idx, gauge_links)
                rel_sum += (cmplx_log(self.compute_plaquette_from_phase( phase_buff )))

                #phase_buff = self.compute_single_plaquette_phase_gen(mu, nu, idx, gauge_links)
                #rel_sum += phase_buff

            Q_values.append(rel_sum.imag)
            #Q_values.append(rel_sum)

        Q_values = np.array(Q_values)
        Q_values /= (2.0*np.pi)

        return Q_values


    def simple_plot(self, vectr, flag_filename, run_params):

        filename = flag_filename + "_vs_MCsteps_" + str(self.latt_size_per_dim[0]) + "_" + str(self.latt_size_per_dim[1]) + ".png"

        x_values = range(len(vectr))

        fig, ax = plt.subplots()
        ax.plot(x_values, vectr)
        ax.grid()
        title = 'Using: '+'mc_steps='+str(run_params['mc_steps'])+', dims='+str(self.latt_size_per_dim) + \
                ', beta='+str(run_params['beta'])
        ax.set(xlabel='MC time', ylabel=flag_filename,
               title=title)
        fig.savefig(filename)
        fig.clf()

        print("Generated " + filename)


    def hist_plot(self, vectr, flag_filename, run_params):

        filename = flag_filename + "_histogram_" + str(self.latt_size_per_dim[0]) + "_" + str(self.latt_size_per_dim[1]) + ".png"

        plt.hist(vectr, bins=40)
        title = 'Using: '+'mc_steps='+str(run_params['mc_steps'])+', dims='+str(self.latt_size_per_dim) + \
                ', beta='+str(run_params['beta'])
        plt.title(title)
        # Save output figure
        plt.savefig(filename)
        plt.clf()

        print("Generated " + filename)


    def compute_avg_plaq(self):

        plaq_values = []

        mu=1
        nu=0
        for i in range(len(self.MARKOV_CHAIN)):

            gauge_links = self.MARKOV_CHAIN[i]

            rel_sum = complex(0.0,0.0)
            for idx in range(self.nr_latt_sites):

                phase_buff = self.compute_single_plaquette_phase_gen(mu, nu, idx, gauge_links)
                rel_sum += self.compute_plaquette_from_phase( phase_buff )

            plaq_values.append(rel_sum)

        plaq_values = np.array(plaq_values)
        plaq_values /= (self.nr_latt_sites)

        return (np.average(plaq_values).real, np.std(plaq_values))


    def clean(self):
        pass




#------------------------------------

    # AUXILIAR FUNCTIONS


    # Q: desired value for the topological charge of the gauge config
    # U: all gauge configs of the simulation
    # i: index of the specific gauge config to be set via Q
    def build_U_with_Q(self, Q):

        #print(self.neighbors_table_forward)
        #print(self.neighbors_table_backward)
        #print(self.neighbors_table_crossed)

        # x direction

        mu=0
        nu=int(not bool(mu))
        for idx in range(self.nr_latt_sites):

            self.gauge_links[idx][mu] = Q * (2.0*np.pi/self.nr_latt_sites) * self.idx_to_coords(idx)[nu]

        # t direction

        mu=1
        nu=int(not bool(mu))
        for idx in range(self.nr_latt_sites):

            t = self.idx_to_coords(idx)[mu]
            if t==(self.latt_size_per_dim[mu]-1):
                self.gauge_links[idx][mu] = -Q * (2.0*np.pi/self.latt_size_per_dim[nu]) * self.idx_to_coords(idx)[nu]
            else:
                self.gauge_links[idx][mu] = 0.0


    def buildM(self, U, c, mass):

        I = []
        J = []
        V = []

        for n in range(self.nr_latt_sites):
            for m in range(self.nr_latt_sites):

                mu=0
                if self.neighbors_table_forward[n][mu] == m:
                    for a in range(2):
                        for b in range(2):
                            i = 2*n+a
                            j = 2*m+b
                            I.append(i)
                            J.append(j)
                            val = 1.0
                            val *= self.gauge_links_exp[n][mu]
                            val *= ( -0.5*self.gamma0[a][b] + c*0.5*self.gammaIdent[a][b] )
                            V.append(val)

                mu=1
                if self.neighbors_table_forward[n][mu] == m:
                    for a in range(2):
                        for b in range(2):
                            i = 2*n+a
                            j = 2*m+b
                            I.append(i)
                            J.append(j)
                            val = 1.0
                            val *= self.gauge_links_exp[n][mu]
                            val *= ( -0.5*self.gamma1[a][b] + c*0.5*self.gammaIdent[a][b] )
                            V.append(val)

                mu=0
                if self.neighbors_table_backward[n][mu] == m:
                    for a in range(2):
                        for b in range(2):
                            i = 2*n+a
                            j = 2*m+b
                            I.append(i)
                            J.append(j)
                            val = 1.0
                            val *= self.gauge_links_exp[self.neighbors_table_backward[n][mu]][mu].conj()
                            val *= ( 0.5*self.gamma0[a][b] + c*0.5*self.gammaIdent[a][b] )
                            V.append(val)

                mu=1
                if self.neighbors_table_backward[n][mu] == m:
                    for a in range(2):
                        for b in range(2):
                            i = 2*n+a
                            j = 2*m+b
                            I.append(i)
                            J.append(j)
                            val = 1.0
                            val *= self.gauge_links_exp[self.neighbors_table_backward[n][mu]][mu].conj()
                            val *= ( 0.5*self.gamma1[a][b] + c*0.5*self.gammaIdent[a][b] )
                            V.append(val)

                if n == m:

                    # only two values to be inserted

                    # 1. first diagonal entry
                    a = 0
                    b = 0
                    i = 2*n+a
                    j = 2*m+b
                    I.append(i)
                    J.append(j)
                    V.append(-mass-c*2.0)


                    # 2. second diagonal entry
                    a = 1
                    b = 1
                    i = 2*n+a
                    j = 2*m+b
                    I.append(i)
                    J.append(j)
                    V.append(-mass-c*2.0)


        I = np.array(I)
        J = np.array(J)
        V = np.array(V)
        tot = self.nr_latt_sites*2
        self.dirac_matrix = sparse.coo_matrix((V,(I,J)), shape=(tot,tot)).tocsr()
