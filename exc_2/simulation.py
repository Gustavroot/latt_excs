# TODO: add a correct way of checking if necessary modules are installed

# Import all necessary modules
import numpy as np
from math import exp, log
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats


# External function to check if pack installed in Python
def check_if_pack_installed(pack_name):
    try:
        return __import__(pack_name)
        #importlib.import_module(full_module_name)
    except ImportError:
        raise Exception("The package "+pack_name+" needs to be installed.")



class Simulation:

    # This will be an array with the values of S_E for each MC time
    S_E = []
    # This will be an array with the values of x for each MC time
    x = []

    # Array to store values of the correlation function
    corrs = []

    # Arrays to store values of the effective energies
    eff_energies_exp = []

    # To use only in the case of the <gaussian_one> proposal distribution
    x_single = []

    # List of names of physical variables of the system
    varnames = ['S_E', 'wave_function', 'eff_energy_exp', 'corrs']

    # List of names of the pdfs available for choosing
    distnames = ['gaussian']
    # List of proposal distributions
    prop_dists = ['gaussian_all', 'gaussian_one']
    # Higher variance leads to a lower acc rate, due to an "unconnected" chain
    gauss_var = 0.1


    # Default constructor
    def __init__(self):
        pass


    def set_params(self, params):
        self.N = params['N']
        self.w = params['w']
        self.burn_in = params['burn_in']
        if 'rnd_seed' in params:
            self.rnd_seed = params['rnd_seed']
            np.random.seed(self.rnd_seed)


    def init_use_prior_pdf(self, distname):
        if distname not in self.distnames:
            raise Exception("Distribution chosen not available.")

        # TODO: generalize the following ---> could be a <PDF> class with multiple methods corresponding
        #       to different pdfs

        if distname=='gaussian':
            self.x.append( np.random.normal(0, self.gauss_var, self.N) )

        self.S_E.append( self.compute_S_E(self.x[0]) )


    def compute_S_E(self, x_p):

        #print(x_p.shape)
        #print(self.N)

        S_E = 0.0
        for i in range(self.N-1):
            S_E += pow(x_p[i+1] - x_p[i], 2)
        S_E += pow(x_p[0] - x_p[self.N-1], 2)
        S_E *= (1.0/self.w)
        
        S_E += ( self.w * pow(np.linalg.norm(x_p), 2) )

        return S_E

    def run_one_step(self, prop_dist, i):
        #print("running one MC step ... ")

        if prop_dist not in self.prop_dists:
            raise Exception("Distribution chosen not available.")

        # TODO: refactor the following if-code, generalize

        if prop_dist=='gaussian_all':

            # 1. generate a candidate x' for the next sample: x' <--- g(x'|x_previous)
            x_p = np.random.normal( self.x[len(self.x)-1], self.gauss_var, self.N )

            # 2. compute the value of S_E for this new x'
            buff_S_E = self.compute_S_E(x_p)
        
            # 3. compute the acceptance ratio
            acc_ratio = exp(-buff_S_E) / exp(-self.S_E[len(self.S_E)-1])
        
            # 4. accept or reject
            u = np.random.uniform(size=1)[0]
            if u <= acc_ratio:
                self.S_E.append( buff_S_E )
                self.x.append( np.copy(x_p) )
            else:
                self.x.append( np.copy( self.x[len(self.x)-1] ) )
                self.S_E.append( self.S_E[len(self.S_E)-1] )

        if prop_dist=='gaussian_one':

            # TODO: the proper way of implementing this case is as follows:
            #		1. instead of copying the whole previous x, store only the newly
            #		   proposed element in the array
            #		2. compute the change in S_E based on the change of this single entry
            #		   of the array
            #		3. when plotting the wave function, use only those values of x_single

            # 1. generate a candidate x' for the next sample: x' <--- g(x'|x_previous)
            #x_p = np.random.normal( self.x[len(self.x)-1], self.gauss_var, self.N )
            x_p = np.copy( self.x[len(self.x)-1] )
            x_p[ i%self.N ] = np.random.normal( self.x[len(self.x)-1][ i%self.N ], self.gauss_var, 1 )

            # 2. compute the value of S_E for this new x'
            buff_S_E = self.compute_S_E(x_p)
        
            # 3. compute the acceptance ratio
            acc_ratio = exp(-buff_S_E) / exp(-self.S_E[len(self.S_E)-1])
        
            # 4. accept or reject
            u = np.random.uniform(size=1)[0]
            if u <= acc_ratio:
                self.S_E.append( buff_S_E )
                self.x.append( np.copy(x_p) )
            else:
                self.x.append( np.copy( self.x[len(self.x)-1] ) )
                self.S_E.append( self.S_E[len(self.S_E)-1] )


    def run(self, mc_steps, prop_dist, delta):
        if mc_steps<=self.burn_in:
            raise Exception("Number of MC steps must be larger than burn-in.")

        for i in range(mc_steps-1):
            self.run_one_step(prop_dist, i)

        self.x = np.array(self.x)
        self.x = self.x[self.burn_in:,:]

        y = []
        for i in range(0, self.x.shape[0], delta):
           y.append(self.x[i,:])

        self.x = y
        self.x = np.array(self.x)


    def compute_corr_fctn(self, n):

        #print(self.x[:,0].shape[0])

        corr = np.dot(self.x[:,0], self.x[:,n]) / self.x[:,0].shape[0]
        return corr


    def compute_corrs(self):
        self.corrs = []
        for i in range(self.N):
            self.corrs.append(self.compute_corr_fctn(i))

        self.corrs = np.array(self.corrs)
        #print(self.corrs)


    def eff_energy_exp_one(self, n):
        energy = log( abs(self.corrs[n] / self.corrs[n+1]) )
        #energy = self.corrs[n] / self.corrs[n+1]

        #energy = np.arccosh([ (self.corrs[n+1] + self.corrs[n-1]) / (2*self.corrs[n]) ])[0]
        #log( abs(self.corrs[n] / self.corrs[n+1]) )

        return energy


    def eff_energy_exp_all(self):

        self.eff_energies_exp = []
        for i in range( (self.N)/2 ):
            self.eff_energies_exp.append(self.eff_energy_exp_one(i))

        self.eff_energies_exp = np.array(self.eff_energies_exp)

        #print(self.eff_energies_exp)


    def plot(self, varname, filename, run_setup):

        if varname not in self.varnames:
            raise Exception("The variable chosen to plot doesn't correspond to a system variable.")
        
        if varname=='S_E':

            fig, ax = plt.subplots()
            ax.plot(range(len(self.S_E)), self.S_E)
            ax.grid()

            ax.set(xlabel='MC time', ylabel='S_E',
                   title='Using: '+'mc_steps='+str(run_setup['mc_steps'])+', N='+str(self.N)+', w='+str(self.w)+', prop dist='+run_setup['prop_dist'])

            fig.savefig(filename)
            fig.clf()

        elif varname=='eff_energy_exp':

            fig, ax = plt.subplots()
            ax.plot(range(len(self.eff_energies_exp)), self.eff_energies_exp)
            ax.grid()

            ax.set(xlabel='t', ylabel='eff energy',
                   title='Using: '+'mc_steps='+str(run_setup['mc_steps'])+', N='+str(self.N)+', w='+str(self.w)+', prop dist='+run_setup['prop_dist'])

            fig.savefig(filename)
            fig.clf()

        elif varname=='corrs':

            #print(range(len(self.corrs)))

            fig, ax = plt.subplots()
            #ax.plot(range(len(self.corrs)), np.log(self.corrs))
            ax.plot(range(len(self.corrs)), self.corrs)
            ax.grid()

            ax.set(xlabel='t', ylabel='corr function',
                   title='Using: '+'mc_steps='+str(run_setup['mc_steps'])+', N='+str(self.N)+', w='+str(self.w)+', prop dist='+run_setup['prop_dist'])

            fig.savefig(filename)
            fig.clf()

        elif varname=='wave_function':

            buff_length1 = len(self.x)

            # Flatten the whole self.x
            self.x = np.ndarray.flatten(self.x)

            # Make a histogram of the simulation-related data
            #plt.hist(self.x[self.burn_in*self.N:], density=True, bins=40)
            plt.hist(self.x, density=True, bins=40)

            # Plot what QM predicts
            x_min = -3.0
            x_max = 3.0
            mean = 0.0
            std = 0.5
            x_QM = np.linspace(x_min, x_max, 1000)
            y_QM = scipy.stats.norm.pdf(x_QM, mean,std)
            plt.plot(x_QM, y_QM, color='coral')

            # Save output figure
            plt.savefig(filename)

            plt.clf()

            # Setting position data to its original shape            
            self.x = np.ndarray.reshape(self.x, (-1, buff_length1))


    def clean(self):
        self.S_E = []
        self.x = []
        self.corrs = []
        self.eff_energies_exp = []
