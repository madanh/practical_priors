import numpy as np
seed = np.random.randint(2**32-1)
print("seed={}".format(seed))
np.random.seed(seed)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
import theano.tensor as tt
import pickle
import srs
# ## Init
# Init the env, load the generated data and true params
import os
import model # this loads the model #hackish but fast
import sample_conf as conf
import util
import preprocess
from shutil import copyfile



K = model.K
# == KNOBS ==
# sampling knobs
chains = conf.chains
tune = conf.tune
nsamples = conf.nsamples_per_chain
njobs = 1

# TODO: make model code pluggable

# Use implicit process based parralllelism
# One process - one chain

# infer chain number from filesystem structure
tracedir_home = 'traces'

if hasattr(conf,'force_chain_number'):
    #if conf.force_chain_number is False :
    curchain = conf.force_chain_number
else: #start sampling in the smallest integer folder without a lock
    curchain = 0
    while curchain<conf.chains:
        tracedir = os.path.join(tracedir_home ,str(curchain))
        if os.path.isdir(tracedir):
            lock_path = os.path.join(tracedir,'madan.lock')
            if os.path.exists(lock_path):
                #This place is occupied
                curchain = curchain +1
                continue
        #This is  an altrenative branch
        break


    if curchain >= chains:
        print('All chains are already present, exiting')
        exit(1)
# create infrastructure for tracking samplers
tracedir = os.path.join(tracedir_home ,str(curchain))
os.makedirs(tracedir,exist_ok=True)
lock_path = os.path.join(tracedir,'madan.lock')
with open(lock_path,'w') as f:
    print(os.getpid(),file = f)
    print("seed="+str(seed),file = f)

## io knobs
# identifier to be appended to filenames to enable multiple data
suffix = ""


# == INFO ==

# np.random.seed(101)
print("numpy version",np.__version__)
print("theano version",theano.__version__)
print("pymc version",pm.__version__)
floatX = theano.config.floatX # "float32"
device = theano.config.device
print("theano.config.device:",device)
if device=='cpu':
    nsamples_aug = nsamples//njobs #how many samples to draw
    nsamples = nsamples_aug*njobs
    print("setting njobs to ",njobs)


# set "environment"
experiment_dir=os.getcwd()+'/'
experiment_id = ''

# paths to generated data
data_file = os.path.join(experiment_dir, experiment_id + suffix + '_data.csv')
# truth_file = os.path.join(experiment_dir, experiment_id + suffix + '_truth')

# Load the data
data = util.load_data(data_file)

M = data.shape[1]
N = data.shape[0]

class Jeff(pm.Continuous):
    """
    Jeffreys prior for scale params
    """
    def __init__(self, *args, **kwargs):
        super(Jeff,self).__init__(*args,**kwargs)

    def logp(self,value):
        return -tt.log(value)


def save_state(trace,tracedir,mvg_model,step,sca=1):
    """

    :param trace:
    :param tracedir:
    :param mvg_model:
    :param step:
    :param sca: sca - an M-vector of scales used to preprocess the data
    :return:
    """
    #THIS IS AD-HOC SPOT ENFORCEMENT
    # varnames shadow global ones
    # TODO: Some day refactor this
    with open(state_path,'wb') as f:
        pickle.dump({'model':mvg_model,
                     'trace':trace,
                     'step':step,
		     'sca':sca},f, protocol=-1)
    clist = []
    for c in trace.chains:
        t = trace._straces[c]
        d = dict()
        for k in t.varnames:
            d[k] = t.get_values(k)
        #Convert chol to SRS(packed) and append it to d:
        SR = np.array([srs.chol_to_srs_packed(cp,M) for cp in d['chol_packed']])
        d['S'] = SR[:,:M]/sca
        d['R'] = SR[:,M:]
        # add q format to trace.pkl:
        # d['q'] = d['x']+model.q0
        clist.append(d)

    tracefile = os.path.join(tracedir,'trace.pkl')
    with open(os.path.join(tracefile),'wb') as f:
        pickle.dump(clist,f)

    # compute and serialize lnp
    lnp = mvg_model.logp
    lp = np.zeros((len(trace.chains),len(trace)))
    for c in trace.chains:
        for i in range(0,len(trace),1):
            lp[c,i] = lnp(trace.point(i,chain=c))

    np.save(os.path.join(tracedir,'lnp.npy'),lp)
    np.savetxt(os.path.join(tracedir,'lnp.csv'),lp)

## GIMEL: SAMPLE
#>>> TUNE <<<
# step = pm.Slice()
state_path = os.path.join(tracedir,'state.pkl')
if os.path.exists(state_path):
    with open(state_path,'rb') as f :
        state = pickle.load(f)
        mvg_model = state['model']
        trace = state['trace']
        sca = state['sca']
        with mvg_model:
            step = state['step']
    if len(trace)>=conf.tune + conf.nsamples_per_chain: #False casts to zero
        print("The saved trace already contains conf.nsamples_per_chain samples")
        print("Exiting")
        exit(0)
else: #De novo sampling:
    # save config
    cfg_name = ['sample_conf.py' ,
                'model.py',
                'sample.py']
    for ff in cfg_name:
        copyfile(ff,os.path.join(tracedir,ff))

    l = model.l
    u = model.u
    print('Order control:')
    print('data[u]:',data[u])
    print('data[l]:',data[l])
    # >>>>> Y RESCALING <<<<<
    if util.conf_set(conf,'rescale'):
        (data,sca,scb) = preprocess.preprocess(data)
    else:
        sca = 1.
        scb = 0.
    xu = model.qu - model.q0
    xl = model.ql - model.q0
    el = model.el
    eu = model.eu

    mvg_model = pm.Model()
    
    # TODO: consider dispensing with testvals e.g. by manually coding ordered transform potential (see forum)
    # testu  = xu
    # testl = xl
    testu  = 1
    if hasattr(conf,'testu'):
        testu  = conf.testu
    testl = -1
    if hasattr(conf,'testl'):
        testu  = conf.testl
    testval = dict()
    if util.conf_set(conf,'from_truth') :
        raise NotImplementedError("Initialization from truth is broken now, need to accound for preprocessing transform")
        with open('_lse.pkl','rb') as f:
            testval= pickle.load(f)
        q_true = {'q':util.load_data('_truth.csv').reshape(-1,1)} # reshape to have (N,1) shape
        testval.update(q_true)
        testval['xn'] = testval['q'] - model.q0
        testval['chol_packed'] = srs.srs_packed_to_chol_packed(testval['S'],testval['R'])
        # testval['S'] = np.max(data.max(axis=0)-data.min(axis=0))/2
        # for k in range(K):
        #     testval['b'+str(k)] = None
    else:
        xn_testval = np.random.uniform(testl,testu,(N,1))
        if xn_testval[u] < xn_testval[l] :
            tmp = xn_testval[l]
            xn_testval[u]=xn_testval[l]
            xn_testval[u] = tmp
        testval['xn'] = xn_testval
        for k in range(K):
            testval['b'+str(k)] = None
        testval['S'] = np.max(data.max(axis=0)-data.min(axis=0))/2
        testval['chol_packed'] = None

    # >>> If you know minimal distance between xu and xl, you can help yourself here
    if util.conf_set(model,'minimal_xu_xl_distance'):
        minimal_xu_xl_distance = model.minimal_xu_xl_distance
    else:
        minimal_xu_xl_distance = 0


    with mvg_model:
        # x_p distribution
        # I tried lower =0 and upper = 1 and t samples)
        # xn = pm.Uniform('xn', lower=testl, upper= testu, shape = (N,1),testval=data[:,0][:,np.newaxis])
        xn = pm.Uniform('xn', lower=testl, upper= testu, shape = (N,1),testval=testval['xn'])
        order_potential = pm.Potential('order_potential', tt.log(tt.nnet.sigmoid((xn[u] - xn[l]-minimal_xu_xl_distance)*conf.order_sigmoid_scale)))
        # order_potential = pm.Potential('order_potential', tt.switch(xn[u]>xn[l],0,-np.inf))
        order_control = pm.Deterministic('order_control',xn[u]-xn[l])
        xpl = pm.Uniform('xpl',lower = xl, upper = xl + el)
        xpu = pm.Uniform('xpu',lower = xu-eu, upper = xu)
        xnl = tt.min(xn)
        xnu = tt.max(xn)

        # linear transform (naive to x_p^prime calculation)
        a1 = pm.Deterministic('a1',(xpu-xpl)/(xnu-xnl)) #TODO: think how we can work around the fact that here we might divide by zero
        a0 = pm.Deterministic('a0',(xpu-a1*xnu))

        x = pm.Deterministic('x',a1*xn+a0)
        q = pm.Deterministic('q',x+model.q0)

        # poly coeffs
        bn = [pm.Flat('bn' + str(k), shape = M, testval = testval['b' + str(k)]) for k in range(K)]
        b = [pm.Deterministic('b0',(bn[0]-scb)/sca)] + \
            [pm.Deterministic('b' + str(k),bn[k]/sca) for k in range(1,K)]

        xxx =pm.math.concatenate([x for _ in range(M)],axis = 1) #that list comprehension is just a "repmat"
        # mu = b0 + \
        #      xxx*b1 + \
        #      xxx*xxx*b2
        mu = bn[0]
        for k in range(1,K):
            mu = mu + bn[k] * xxx ** k

        # covariance parametrization
        sd_template = pm.Bound(Jeff,lower = 0.01, upper = np.max(data.max(axis=0)-data.min(axis=0)))
        sd_dist = sd_template.dist(shape=M,testval=testval['S'])
        # sd_dist = pm.HalfCauchy.dist(1.,shape = M,testval = 1.,)
        # sd_dist = pm.Uniform.dist(lower = 0., upper = data.max(axis=0)-data.min(axis=0), shape = M)
        chol_packed = pm.LKJCholeskyCov('chol_packed', n=M, eta = 1.,sd_dist=sd_dist,testval = testval['chol_packed'])
        chol = pm.expand_packed_triangular(M,chol_packed)
        # data connection
        y = pm.MvNormal('y',mu = mu, chol= chol, shape = (N,M),observed = data)
        # mvg_model.logp()


    with mvg_model:
        if conf.approx:
            approx = pm.fit(method=conf.approx)
            trace = approx.sample(conf.nsamples_per_chain)
            save_state(trace,tracedir,mvg_model,approx,sca)
            exit(0)

        step = pm.NUTS()
        if util.conf_set(conf,'from_truth'):
            start_ = None
        else:
            # n_init = 2000000
            #random_seed = 1337

            # start_, step = pm.sampling.init_nuts(init=conf.init, chains=1, n_init=n_init,
            #                          model=model, random_seed=random_seed,
            #                          progressbar=True,)
            start_, step = pm.sampling.init_nuts(init=conf.init, chains = 1)#, n_init=n_init)

            # if conf_set('debug'):
            #     plt.plot(data[:, 0], start_[0]['xn'], 'ro')
            #     plt.gcf().savefig(os.path.join(tracedir,'init_x_vs_data_0.png'))
        #     # plt.show()

        trace = pm.sample(draws=conf.block_size,
                          chains=1,
                          njobs=njobs,
                          discard_tuned_samples=conf.discard_tuned_sample,
                          tune=tune,
                          init =conf.init,
                          step=step,
                          start = start_,
                          )
    save_state(trace,tracedir,mvg_model,step,sca)

    #>>> MINE BLOCKS <<<
while len(trace)<conf.nsamples_per_chain+conf.tune:
    with mvg_model:
        trace = pm.sample(draws=conf.block_size,
                           trace =trace,
                           init=None,
                           chains=1,
                           njobs=njobs,
                           tune=False,
                          step=step)
    # dump a block
    save_state(trace,tracedir,mvg_model,step,sca)
# remove lock
# commenting this for convenience of rapid sampling
# TODO: fix this
# os.remove(lock_path)

## HE: VISUALIZE
_ = pm.traceplot(trace,combined=True)
plt.gcf().savefig(os.path.join(tracedir,'traceplot.png'))
# plt.show()

pass
