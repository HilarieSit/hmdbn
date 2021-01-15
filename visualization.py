import matplotlib.pyplot as plt
import numpy as np

def plot_posteriors(P, configs):
    plt.clf()
    c, _ = P.shape
    if type(configs) != list:
        configs = [configs]
    for curve in range(c):
        plt.plot(np.linspace(1, 65, 65), P[curve,:])
    parents = []
    for config in configs:
        parents.append([parents.gene for parents in config.parents])
    plt.title(parents)
    plt.legend(parents)
    plt.show()