import numpy as np
import model
import srs
import pickle
import lse_conf as conf
import util

true_q_filename = '_truth.csv' #TODO: harmonize with suffix convention
data_filename = '_data.csv'
lse_output_filename = '_lse.pkl'

# Read in q from _truth
q = np.loadtxt(true_q_filename)

# Read in q0 and K from model
q0 = model.q0
K = model.K

y = util.load_data(data_filename)

M = y.shape[1]

# Fit polynomials
x = q - q0
# Fit the data to xt
## polynomial fitting as per http://stackoverflow.com/a/19165440/3791466
b= [None]*M  # init the array with right number of els
res= [None]*M
for m in range(0, M):
    y_m = y[:, m]
    b[m], diag = np.polynomial.polynomial.polyfit(x, y_m, K-1, full=True)
    # get residuals
    res[m] = np.polynomial.polynomial.polyval(x,b[m])-y_m

# Get covm
covm = np.cov(res,ddof=K)

# Prepare for marshalling (pickle)
out = dict()
# Format b as b_k
b = np.array(b)
for k in range(K):
    out['b'+str(k)] = b[:,k]


# Format srs as sigma , R[triu]
(Sp,Rp) = srs.cov_to_srs_packed(covm)
out['S'] = Sp
out['R'] = Rp

# Save
with open(lse_output_filename,'wb') as f:
    pickle.dump(out,f)

# Optional: make a picture
def polyfitvis(b,q,y,q0,ql=None,qu=None,save = True, show = True):
    if ql is None:
        ql = q.min()
    if qu is None:
        qu = q.max()

    M = y.shape[1]

    qc = np.linspace(ql,qu,1000)# q continuous
    xc = qc-q0
    for m in range(M):
        yml = np.polynomial.polynomial.polyval(xc,b[m])
        fig, ax = plt.subplots()
        plt.plot(qc,yml)
        plt.plot(q,y[:,m],'k.')
        # save the figure (optional)
        if save:
            fig.savefig('ls_fit_'+str(m)+'.png')
    # show the figure (optional)
    if show:
        plt.show()
    return fig

visshow = conf.visshow
vissave= conf.vissave

if visshow or vissave:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    polyfitvis(b,q,y,q0,save = vissave, show = visshow)
