import numpy as np

from cmath import exp as cmplx_exp
from math import cos, exp


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


    # this method returns S^{gauge}_{E} / \beta
    def compute_gauge_action(self):

        action = 0.0
        for idx in range(self.nr_latt_sites):
            for nu in range(self.spacetime_dim):
                for mu in range(nu):

                    #buff_nr = complex(1.0,0.0) - self.compute_plaquette_from_phase( self.compute_single_plaquette_phase(mu,nu,idx) )
                    #buff_nr = buff_nr.real

                    # take real part directly
                    buff_nr = 1.0 - cos( self.compute_single_plaquette_phase(mu,nu,idx) )

                    action += buff_nr

        return action


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
        print("")
        print("Parameters of the run:")
        print("\t beta: "+str(beta))
        print("\t mc_steps: "+str(mc_steps))
        print("\t burn_in: "+str(burn_in))
        print("\t skip: "+str(skip))
        print("\t effective number of 'independent' configurations: "+str((mc_steps-burn_in)/skip))

        if burn_in>mc_steps:
            raise Exception("ERROR: burn_in can't be greater than mc_steps.")

        # burn-in
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

        self.MARKOV_CHAIN = []

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
