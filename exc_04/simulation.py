import numpy as np

from math import exp, log, acosh
import matplotlib.pyplot as plt
import sys


"""

(1+1)-dimensional structure:

 --------------< temporal (self.n_t), positive downwards << direction 2-hat >>
|
|
|
|
|
|
^
spatial (self.n_s), positive to the right << direction 1-hat >>

"""

class Simulation:

    phi = []
    phi_buff = []
    S_E = []
    # phi in momentum, corresponding to a FT of phi
    phi_momentum = []
    corrs = []
    eff_energies = []

    def __init__(self, params):
        self.n_s = params['n_s'] #horizontal direction ('columns')
        self.n_t = params['n_t'] #vertical direction
        self.m = params['m']

        # random seed
        np.random.seed(params['rnd_seed'])

        self.gauss_var = 0.1

        # Initial set of field values through Gaussian prior distribution
        self.phi_buff = np.random.normal(0, self.gauss_var, (self.n_s, self.n_t))


    def compute_delta_action(self, phi, phi_val_prime, x, t):

        phi_val = phi[x,t]
        
        delta_phi = (phi_val_prime - phi_val)

        term1 = ( pow(phi_val_prime,2)-pow(phi_val,2) ) * (pow(self.m,2) + 4.0)
        term2 = phi[(x+1)%self.n_s,t]
        term3 = phi[x,(t+1)%self.n_t]
        term4 = phi[(self.n_s+x-1)%self.n_s,t]
        term5 = phi[x,(self.n_t+t-1)%self.n_t]

        delta_S_E = term1 - 2*delta_phi*(term2+term3+term4+term5)

        # return S_E' - S_E (new - old)
        return 0.5*delta_S_E


    def compute_action(self, phi):
        return 0


    def run_one_step(self, x, t):
        
        # 1. proposal
        phi_prop = np.random.normal( self.phi_buff[x,t], self.gauss_var, 1 )[0]
        
        # 2. compute the change in action
        delta_S_E = self.compute_delta_action( self.phi_buff, phi_prop, x, t )

        # 3. acceptance ratio
        acc_ratio = exp( -delta_S_E )

        # 4. accept or reject
        #self.buff_phi.append( np.copy(self.buff_phi[len(self.buff_phi)-1]) )
        u = np.random.uniform(size=1)[0]
        if u <= acc_ratio:
            self.phi_buff[x,t] = phi_prop


    def run(self, params):
        burn_in = params['burn_in']
        mc_steps = params['mc_steps']
        skip_length = params['skip_length']

        print("\nBurn-in...")

        # run thermalization, discard burn-in
        for i in range(burn_in):
            t = i%self.n_s
            x = (i/self.n_s)%self.n_t
            self.run_one_step(x,t)

        print("...done.")

        # overall counter
        w = 0

        progr_prev = 0
        while w<(mc_steps-burn_in):

            # even
            for i in range(0,self.n_t*self.n_s,2):

                if w>=(mc_steps-burn_in): break

                t = i%self.n_t
                x = (i/self.n_t)%self.n_s
                self.run_one_step(x,t)

                # print progress
                progr = int((float(w) / float(mc_steps-burn_in))*100.0)
                if progr>0 and progr!=progr_prev and progr%2==0:
                    print(str(progr)+"% ..."),
                    sys.stdout.flush()
                    progr_prev = progr

                # save configs
                if w%skip_length==0:
                    self.phi.append( np.copy(self.phi_buff) )

                w+=1

            # odd
            for i in range(1,self.n_t*self.n_s,2):

                if w>=(mc_steps-burn_in): break

                t = i%self.n_t
                x = (i/self.n_t)%self.n_s
                self.run_one_step(x,t)

                # print progress
                progr = int((float(w) / float(mc_steps-burn_in))*100.0)
                if progr>0 and progr!=progr_prev and progr%2==0:
                    print(str(progr)+"% ..."),
                    sys.stdout.flush()
                    progr_prev = progr

                # save configs
                if w%skip_length==0:
                    self.phi.append( np.copy(self.phi_buff) )

                w+=1

        self.phi = np.array(self.phi)


    # Fourier Transform
    def apply_ft(self):
        y = []
        for i in range(self.phi.shape[0]):
            y.append( [] )
            # for a fixed config, run over t values (j takes values of time t)
            for j in range(self.phi.shape[2]):
                #if i==0 and j==0:
                #    print(np.fft.fft( self.phi[i,:,j] ))
                y[len(y)-1].append( np.fft.fft( self.phi[i,:,j] ) )

        self.phi_momentum = y
        self.phi_momentum = np.array(self.phi_momentum)

        # re-organize data, to have the same structure of access [config,x,t] ---> [config,p,t]
        for i in range(self.phi_momentum.shape[0]):
            self.phi_momentum[i] = np.transpose(self.phi_momentum[i])


    def compute_corrs(self):

        # run over values of p
        for k in range(self.phi_momentum.shape[1]):
            self.corrs.append( [] )
            # run over values of time t
            for j in range(self.phi_momentum.shape[2]):

                self.corrs[len(self.corrs)-1].append( np.vdot( self.phi_momentum[:,k,0], self.phi_momentum[:,k,j] ) / self.phi_momentum[:,k,0].shape[0] )

        self.corrs = np.array(self.corrs)
        #print("")
        #print(self.corrs)

        # TODO: not necessary?
        #self.corrs = np.absolute(self.corrs)
        self.corrs = np.absolute( np.real(self.corrs) )


    def compute_eff_energies(self, eff_en_type):

        for k in range(self.corrs.shape[0]):
            self.eff_energies.append( [] )
            for j in range(self.corrs.shape[1]-1):
            #for j in range(self.corrs.shape[1]/2):

                if eff_en_type=="exp":
                    self.eff_energies[len(self.eff_energies)-1].append( log( self.corrs[k,j]/self.corrs[k,j+1] ) )
                elif eff_en_type=="3pt":
                    self.eff_energies[len(self.eff_energies)-1].append( acosh( (self.corrs[k,j+1]+self.corrs[k,j-1]) / (2*self.corrs[k,j]) ) )
                else:
                    raise Exception("Effective energy type not available.")

        self.eff_energies = np.array(self.eff_energies)


    def plot(self, varname, filename, run_setup):

        if varname=="corrs":
            var_to_plot = self.corrs[1,:]
        elif varname=="eff_energies":
            var_to_plot = self.eff_energies[1,:]
        else:
            raise Exception("Plotter not enabled for the chosen variable.")

        fig, ax = plt.subplots()
        ax.plot(range(len(var_to_plot)), var_to_plot)
        ax.grid()

        ax.set(xlabel='MC time', ylabel=varname,
               title='Using: '+'mc_steps='+str((run_setup['mc_steps']-run_setup['burn_in'])/run_setup['skip_length']) \
               +', n_s='+str(self.n_s)+', n_t='+str(self.n_t)+', m='+str(self.m))

        fig.savefig(filename)
        fig.clf()


    def clean(self):
        pass
