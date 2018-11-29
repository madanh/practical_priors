
import numpy as np
import pickle
import generate_conf as t
import model

if hasattr(t,'seed'):
    np.random.seed(t.seed)

options = dict()
options['show'] = True

b_t = np.array(t.b) # response polynomial coeffs, per method
if b_t.shape[1] != model.K:
    raise ValueError("b matrix in generate_conf must have number of colums equal to K from model.py")
M = b_t.shape[0] # number of methods
N = t.N # number of points/patients
q = np.random.uniform(model.ql, model.qu, N)
# respect u and l:
if q[model.u]<q[model.l]:
    tmp = q[model.u]
    q[model.u]=q[model.l]
    q[model.l]=tmp
# respect model.eu:
pmax = np.argmax(q)
if q[pmax]<model.qu-model.eu:
    q[pmax] = np.random.uniform(model.qu-model.eu,model.qu)
# respect model.el:
pmin = np.argmin(q)
if q[pmin]>model.ql+model.el:
    q[pmin] = np.random.uniform(model.ql,model.ql+model.ql)
if hasattr(t,'force_distance'):
    if t.force_distance:
        q[model.l] = np.random.uniform(model.ql,model.ql+model.ql)
        q[model.u] = np.random.uniform(model.qu-model.eu,model.qu)


x_t = q-model.q0
mean_t = np.zeros(M)
# S_t = np.eye(M)*0.01
S_t = np.diag(t.S)
R_t = np.array(t.R)
cov_t = np.dot(np.dot(S_t, R_t), S_t)
x_pm = np.polynomial.polynomial.polyval(x_t, b_t.T)
data = np.random.multivariate_normal(mean=mean_t, cov=cov_t, size=N) + x_pm.T


if options['show']:
    import matplotlib as mpl
    mpl.use('Agg') # supress demands for graphical display
    import matplotlib.pyplot as plt

    for m in range(M):
        plt.scatter(x_t, data[:,m])
    plt.gcf().savefig('gen.png')
    # plt.show()

np.savetxt('_data.csv',data)

# save "truth" in q format (according to the standard)
np.savetxt('_truth.csv', q)
