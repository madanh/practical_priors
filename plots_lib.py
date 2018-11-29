"""
This is a file for all plotting functions in the analyse.py. This is to improve readability
and force me to write a more modular code.
"""


def trace(figsize = (40,30)):
    fig, ax = plt.subplots(figsize=figsize)
    y = np.vstack((t[k].squeeze() for t in trace))
    plt.plot(y)
    plt.legend(list(range(y.shape[1])))
    plt.title(k)
    js(fig,k)
