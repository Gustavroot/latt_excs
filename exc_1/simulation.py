import numpy as np
from math import exp

import matplotlib
import matplotlib.pyplot as plt

class Simulation:

    # This will be an array with the values of S_E for each MC time
    S_E = []
    # This will be an array with the values of x for each MC time
    x = []

    # List of names of physical variables of the system
    varnames = ['S_E']

    # List of names of the pdfs available for choosing
    distnames = ['gaussian']
    # Higher variance leads to a lower acc rate, due to an "unconnected" chain
    gauss_var = 0.1


    # Default constructor
    def __init__(self, params):
        self.N = params['N']
        self.w = params['w']
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
        S_E = 0.0
        for i in range(self.N-1):
            S_E += pow(x_p[i+1] - x_p[i], 2)
        S_E += pow(x_p[0] - x_p[self.N-1], 2)
        S_E *= (1.0/self.w)
        
        S_E += ( self.w * pow(np.linalg.norm(x_p), 2) )

        return S_E

    def run_one_step(self, prop_dist):
        #print("running one MC step ... ")

        if prop_dist not in self.distnames:
            raise Exception("Distribution chosen not available.")

        # TODO: refactor the following if-code, generalize

        if prop_dist=='gaussian':

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
                #print("SAME")


    def run(self, mc_steps, prop_dist):
        #print("overall simulation:")
        
        for i in range(mc_steps):
            self.run_one_step(prop_dist)
            #print(self.x)


    def plot(self, varname, filename, run_setup):

        if varname not in self.varnames:
            raise Exception("The variable chosen to plot doesn't correspond to a system variable.")
        
        fig, ax = plt.subplots()
        
        ax.plot(range(len(self.S_E)), self.S_E)
        
        ax.grid()

        ax.set(xlabel='MC time', ylabel='S_E',
               title='Using: '+'mc_steps='+str(run_setup['mc_steps'])+', N='+str(self.N)+', w='+str(self.w)+', prop dist='+run_setup['prop_dist'])

        fig.savefig(filename)
        
        fig.clf()
