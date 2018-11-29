import numpy as np
import matplotlib.pyplot as plt
def load_data(data_filename):
    # Read in y_pm from _data
    try:
        y = np.loadtxt(data_filename)
    except ValueError:
        y = np.loadtxt(data_filename,delimiter=',')
    return y

def conf_set(conf,opt):
    """
    Verify if config file has given option set and true
    :param opt:
    :return:
    """
    if hasattr(conf,opt):
        if getattr(conf,opt):
            return True
    return False

def getdirlist(dir):
    return sorted(next(os.walk(dir))[1])

def add_identity_lines():
    plt.gca().set_aspect('equal')
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    l = min(min(xlim),min(ylim))
    h = max(max(xlim),max(ylim))
    plt.plot((l,h),(l,h),'k--')
    plt.gca().set_aspect('equal')
