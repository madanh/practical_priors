import numpy as np
import  pickle
# TODO: relegate this ot config when it grows large
u = 1.
l = -1.
ax = 0

def preprocess(data, l = l, u = u,state_filename = None, ax = ax):
    """
    rescale the data, so that NUTS and ADVI feel more comfortable
    :param data:
    :param l:
    :param u:
    :param state_filename:
    :return:
    """
    du = data.max(axis = ax)
    dl = data.min(axis = ax)
    alpha = (u-l)/(du - dl)
    beta = l - alpha*dl

    #serialize
    if state_filename:
        out = {"alpha":alpha,"beta":beta}
        with open(state_filename,'wb') as f:
            pickle.dump(out,f)

    return (alpha*data + beta,alpha,beta)