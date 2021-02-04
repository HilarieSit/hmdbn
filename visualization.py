import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from hmdbn import *

'''
Plot posterior distribution for dictionary of HMDBNs and save png
Arguments:
    hmdbns [list]: hmdbns to plot
'''
def plot_posteriors(hmdbns):
    font = {
        'size': 14}

    matplotlib.rc('font', **font)   
    plt.clf()
    n_subplots = len(hmdbns)
    fig, axs = plt.subplots(n_subplots, figsize=(10,20))
    axs[0].set_title('Posterior distribution of HMDBNs')

    for i, (gene, model) in enumerate(hmdbns.items()):
        P = model.posterior
        hid_states = [str(states).replace('[]', 'no parents').replace('[','').replace(']','').replace('\'','') for states in model.hid_states]
        c, _ = P.shape
        for curve in range(c): 
            axs[i].plot(np.linspace(1, 65, 65), P[curve,:])

        axs[i].axvspan(1, 30, alpha=0.05, color='red')                                  # embryonic
        axs[i].axvspan(30, 40, alpha=0.05, color='green')                               # larval
        axs[i].axvspan(40, 58, alpha=0.05, color='blue')                                # pupal
        axs[i].axvspan(58, 65, alpha=0.05, color='purple')                              # adult

        axs[i].legend(hid_states, bbox_to_anchor=(1, 1.05), loc='upper left')
        axs[i].get_xaxis().set_visible(False)
        axs[i].set_ylabel(model.gene)
        axs[i].set_xlim([1, 65])
 
    axs[i].get_xaxis().set_visible(True)
    axs[i].set_xlabel('time points')
    fig.tight_layout()
    fig.savefig('posterior.png')
    plt.show()

if __name__ == '__main__':
    hmdbns = load_hmdbns('models/final_hmdbns')
    plot_posteriors(hmdbns)

