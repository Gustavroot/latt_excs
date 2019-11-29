import numpy as np

from cmath import exp as cmplx_exp


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
    neighbors_table = []


    def __init__(self, params):

        self.spacetime_dim = params['spacetime_dim']

        if len(params['latt_size_per_dim']) != self.spacetime_dim:
            raise Exception("Check your input parameter 'latt_size_per_dim'.")
        else:
            self.latt_size_per_dim = np.array(params['latt_size_per_dim'])

        # create empty NumPy array (virtual memory advantages kick-in)
        self.gauge_links = np.empty([np.prod(self.latt_size_per_dim),self.spacetime_dim])
        self.gauge_transf = np.empty([np.prod(self.latt_size_per_dim),1])
        self.neighbors_table = np.empty([np.prod(self.latt_size_per_dim),self.spacetime_dim], dtype='int')

        np.random.seed(params['rnd_seed'])

        # initialize gauge links
        self.set_gauge_links(params['gauge_links_init'])
        
        # pre-build table with indices of neighbors
        self.build_neighbors_table()


    # <init_mode> \in {'ones','random'}
    def set_gauge_links(self, init_mode):

        if init_mode == 'ones':
            # for gauge links equal to 'ones', their phases must be 'zeroes'
            self.gauge_links.fill(0.0)
        elif init_mode == 'random':
            self.gauge_links = np.random.uniform(0,2.0*np.pi,self.gauge_links.shape)
        else:
            raise Exception("Chosen init mode for gauge links not available.")


    def build_neighbors_table(self):

        for idx in range(np.prod(self.latt_size_per_dim)):
            for mu in range(self.spacetime_dim):
                # convert idx into lattice coordinates
                coords_n = self.idx_to_coords(idx)
                # get index for n+mu
                coords_n_p_mu = np.copy(coords_n)
                coords_n_p_mu[mu] += 1
                # boundary conditions
                coords_n_p_mu[mu] = coords_n_p_mu[mu]%self.latt_size_per_dim[mu]
                idx_n_p_mu = self.coords_to_idx(coords_n_p_mu)
                self.neighbors_table[idx][mu] = idx_n_p_mu


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
        idx_n_p_mu = self.neighbors_table[idx][mu]
        plaq_value += self.gauge_links[idx_n_p_mu][nu]
        # daggered entries
        idx_n_p_nu = self.neighbors_table[idx][nu]
        plaq_value -= self.gauge_links[idx_n_p_nu][mu]
        plaq_value -= self.gauge_links[idx][nu]

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

        idx_n_p_mu = self.neighbors_table[idx][mu]

        phase = 0.0
        phase += self.gauge_transf[idx][0]
        phase += self.gauge_links[idx][mu]
        # daggered term
        phase -= self.gauge_transf[idx_n_p_mu][0]

        return phase


    # this method returns S^{gauge}_{E} / \beta
    def compute_gauge_action(self):

        action = 0.0
        for idx in range(np.prod(self.latt_size_per_dim)):
            for nu in range(self.spacetime_dim):
                for mu in range(nu):
                    buff_nr = complex(1.0,0.0) - self.compute_plaquette_from_phase( self.compute_single_plaquette_phase(mu,nu,idx) )
                    buff_nr = buff_nr.real
                    action += buff_nr

        return action


    def full_gauge_transf(self):
        for i in range(np.prod(self.latt_size_per_dim)):
            for mu in range(self.spacetime_dim):
                # compute the transformed phase
                changed_phase = self.gauge_transf_single_link(mu,i)
                # change the corresponding link
                self.gauge_links[i,mu] = changed_phase


    def run(self):
        pass




    def clean(self):
        pass
