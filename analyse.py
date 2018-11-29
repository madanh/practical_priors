import re
from uncertainties import ufloat
import numpy as np
import model
import pickle
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
# the following two liones from https://stackoverflow.com/a/17390833/3791466
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
# the following knobs are from https://stackoverflow.com/a/39566040/3791466
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
import os
import util
# import plots_lib
import scipy.stats as st
import analyse_conf as conf
from shutil import copyfile
import seaborn as sns

def main(output_dir="analysis/images/", chains=None):
    if not (chains is None):
        conf.chains = chains
    os.makedirs(output_dir, exist_ok=True)
    def js(fig,name):
        fig.savefig(os.path.join(output_dir, name) + '.png',dpi=600)
        plt.close(fig)
    ## save config
    cfg_name = ['analyse_conf.py' ,
                'analyse.py']
    for ff in cfg_name:
        copyfile(ff,os.path.join(output_dir,ff))

    ## default fig sizes
    square_fig_size = (10,10)
    if util.conf_set(conf,'square_fig_size'):
        square_fig_size = conf.square_fig_size

    fig_size = (20,15)
    if util.conf_set(conf,'fig_size'):
        fig_size = conf.square_fig_size

    ## Load the requested chains
    tracedir = 'traces'
    trace = dict()
    for c in conf.chains:
        fname = os.path.join(tracedir,str(c))
        fname = os.path.join(fname,'trace.pkl')
        with open(fname, 'rb') as f:
            trace[c] = pickle.load(f)[0] # This assumes that there is only one chain per file

    # TODO: [C] enable "all" option below
    # if conf.chains=="all":
    #     conf.chains = list(range(len(trace)))


    #>>>>> TRACEPLOTS <<<<<
    if conf.trace_vars:
        for k in conf.trace_vars:
            fig, ax = plt.subplots(figsize=fig_size)
            y = np.vstack((trace[c][k][conf.burn:conf.end:conf.thin].squeeze() for c in conf.chains))
            if y.shape[0]==1:
                y=y.T
            plt.plot(y)
            plt.legend(list(range(y.shape[1])))
            plt.title(k)
            try:
                plt.tight_layout()
            except ValueError:
                print("Tight layout failed (matplotlib bug), ignoring and proceeding...")
            js(fig,'thinned_trace_'+k)

    #>>>> Histograms <<<<
    if hasattr(conf,'hist_vars'):
        if conf.hist_vars:
            for k in conf.hist_vars:
                fig, ax = plt.subplots(figsize=fig_size)
                y = np.vstack((trace[t][k][conf.burn:conf.end:conf.thin].squeeze()
                                for t in trace))
                if y.shape[0]==1:
                    y=y.T
                # for i in y.T:
                #     plt.hist(i,20,histtype='step',normed = True)
                for i in range(y.shape[1]):
                    sns.kdeplot(y.T[i],label=k+str(i),shade=True)
                    m = y.T[i].mean()
                    h = np.percentile(y.T[i], 95) # , axis = 0)
                    l = np.percentile(y.T[i],  5) # , axis = 0)
                    print('{} = {} +/- {}'.format(k+str(i),m, h-l))
                plt.legend()
                # plt.legend(list(range(y.shape[1])))
                plt.yticks([])
                plt.title(k)
                try:
                    plt.tight_layout()
                except ValueError:
                    print("Tight layout failed (matplotlib bug), ignoring and proceeding...")
                js(fig,'hist'+k)
                # if k=='q':
                if True:
                    count = 0
                    for i in y.T:
                        fig, ax = plt.subplots(figsize=fig_size)
                        plt.hist(i,20,histtype='step')
                        plt.tight_layout()
                        js(fig,k+str(count))
                        count += 1



    #>>>> Predictives on pseudo_q <<<<
    # These are dependent on a pre-set set of params, the only DoF is the truth source
    def predictive_curves(q_source = "trace", extra_colors = None):
        """

        :param q_source:
        :param extra_colors: if specified will add colors for nonnegative numbers of this list/array
        :return:
        """
        """
        Make predictive graphs
        :param q_source: {trace|truth} Take q_values from hidden parameter estimates (trace) or gold standard file (truth)
        :return:
        """
        if q_source != "truth":
            qagg = np.vstack(
                (trace[c]['q'][conf.burn:conf.end:conf.thin].squeeze() for c in conf.chains)
            )
            q = qagg.mean(axis = 0)
            qlabel = "$\\tilde{q}$"
        else:
            q = truth['q']
            qlabel = "q"

        if hasattr(conf,'opacity'):
            opacity = conf.opacity
        else:
            opacity = 1.
        ncurves = 200
        K = model.K
        q_pred = np.linspace(model.ql,model.qu,)

        data_filename = '_data.csv'
        # Read in y_pm from _data
        y = util.load_data(data_filename)

        M = y.shape[1]
        alpha = min(1,opacity/ncurves)
        for m in range(M):
            chosen = np.hstack((
                np.vstack((
                    trace[c]['b'+str(k)][conf.burn:conf.end:conf.thin,m]
                    for c in conf.chains)).T
                for k in range(K)))
            chosen_s =np.hstack((
                trace[c]['S'][conf.burn:conf.end:conf.thin,m].squeeze() for c in conf.chains
            ))
            ncurves = min(ncurves,chosen.shape[0])
            choice = np.random.choice(chosen.shape[0],ncurves,replace=0)
            chosen = chosen[choice,:]
            chosen_s = chosen_s.squeeze()[choice]
            fig,ax = plt.subplots(figsize = conf.fig_size)
            for i in range(len(choice)):
                # for b in chosen: # b.shape must be (1,K)
                y_pred = np.polynomial.polynomial.polyval(q_pred-model.q0,chosen[i])
                s = chosen_s[i]
                ax.fill_between(q_pred,y_pred-s,y_pred+s,color = 'k', alpha = alpha)
                #plt.plot(q_pred,y_pred,'r', linewidth = chosen_s[i], alpha = min(1,opacity/ncurves))
                #plt.plot(q_pred,y_pred,'k', alpha = alpha)
                # TODO: width of the curves proportional to sigma
            plt.scatter(q,y[:,m],24,'k',marker='o')
            plt.scatter(q,y[:,m],6,'w',marker='o')

            ## This fires if you supply '_classifications.csv'
            if not (extra_colors is None):
                ## This assumes only 1 and 0 marks are in extra_colors
                tp = np.where(extra_colors==1)[0]
                tn = np.where(extra_colors==0)[0]
                plt.scatter(q[tp],y[tp,m],6,'m',marker='o')
                plt.scatter(q[tn],y[tn,m],6,'c',marker='o')

            ## highlight order disambiguation pair
            u = model.u
            l = model.l
            plt.scatter(q[l],y[l,m],6,'g',marker='o')
            plt.scatter(q[u],y[u,m],6,'r',marker='o')

            #plt.show()
            plt.title("m="+str(m))
            ax.set_xlabel(qlabel)
            ax.set_ylabel('y')
            plt.tight_layout()
            if hasattr(conf,'genuine_mm')&conf.genuine_mm:
                util.add_identity_lines()
            js(fig, 'pred_m_eq_'+str(m)+'_'+q_source)

    classif_file = '_classifications.csv'
    if os.path.exists(classif_file ):
        extra_colors = np.genfromtxt(classif_file,delimiter=',',missing_values='',filling_values=-1.0,
                      usemask=False,invalid_raise=False,)
    else:
        extra_colors = None

    predictive_curves(q_source="trace",extra_colors = extra_colors)


    #>>>> Corners for diag <<<<

    #======================================================================
    #  GRAPHS FOR VALIDATION: WORK ONLY WHEN TRUTH IS AVAILABLE
    #======================================================================

    # TODO: >>>> SEPARATE VALIDATION AND REAL LIFE USE <<<< (Ad hoc solutoin below)
    truth = dict()
    try:
        truth['q'] = np.loadtxt('_truth.csv')
    except IOError:
        print('Truth not available (truth.csv not found)')
        print('Returning')
        return


    if conf.source == 'lse':
        lse_output_filename = '_lse.pkl'
        with open(lse_output_filename,'rb') as f:
            lse = pickle.load(f)
        truth.update(lse)
        xlabel = 'LS estimates'
        qxlabel = 'Reference values'
    elif conf.source == 'gen':
        import generate_conf as genconf
        b = np.array(genconf.b)
        for k in range(b.shape[1]):
            truth['b'+str(k)] = b[:,k]
        truth['S'] = np.array(genconf.S)
        R = np.array(genconf.R)
        truth['R'] = R[np.triu_indices_from(R,1)]
        xlabel = 'True values'
        qxlabel = xlabel


    def get_yagg(v):
        yagg = np.vstack( (trace[c][v][conf.burn:conf.end:conf.thin].squeeze() for c in conf.chains))
        if yagg.shape[0]==1:
            yagg = yagg.T
        return  yagg

    def add_identity_line(ax, x_only = False):
        x = np.array(ax.get_xlim())
        y = np.array(ax.get_ylim())
        if x_only:
            xmax = x.max()
            xmin = x.min()
        else:
            xmax = np.maximum(x,y).max()
            xmin = np.minimum(x,y).min()
        if xmin==xmax:
            xmin = 0
            xmax = 1
        ax.plot((xmin,xmax),(xmin,xmax),'k--')


    #>>>>> VERSUS <<<<<<
    def plot_verus(v,yagg,x,suppress_aspect = False):
        m = yagg.mean(axis = 0)
        h = np.percentile(yagg, 95, axis = 0)
        l = np.percentile(yagg, 5, axis = 0)
        fig,ax = plt.subplots(figsize = square_fig_size)
        #plt.plot(x,m,'k.')
        ax.errorbar(x,m,[m-l,h-m],fmt = 'ko',alpha = 0.7)
        if suppress_aspect:
            ax.set_aspect('auto','datalim')
        else:
            ax.set_aspect('equal','datalim')
        add_identity_line(ax,x_only=suppress_aspect)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Estimates')
        if v=='q':
            #highlight
            u = model.u
            lo = model.l #lo, since l is taken
            plt.scatter(x[lo],m[lo],6,'g',marker='o')
            plt.scatter(x[u],m[u],6,'r',marker='o')
            ax.errorbar(x[u],m[u],[[m[u]-l[u]],[h[u]-m[u]]],fmt = 'ro',alpha = 0.7)
            ax.errorbar(x[lo],m[lo],[[m[lo]-l[lo]],[h[lo]-m[lo]]],fmt = 'go',alpha = 0.7)
            #Set human-supplied name
            if util.conf_set(conf,'qname'):
                tit = conf.qname
            else:
                tit = ""
            ax.set_xlabel(qxlabel)
            plt.title(tit)
        else:
            plt.title(v)
        plt.tight_layout()
        js(fig,'versus_'+v)
    if conf.versus_vars:
        for v in conf.versus_vars:
            yagg = get_yagg(v)
            x = truth[v]
            plot_verus(v,yagg,x)

    #>>>>> VERSUS DENSITY <<<<<
    if conf.versus_density_vars:
        for v in conf.versus_density_vars:
            yagg = get_yagg(v)
            # m = yagg.mean(axis = 0)
            x = truth[v]
            fig,ax = plt.subplots(figsize = square_fig_size)
            visthin = max(1,yagg.shape[0]//800) #have approx 500 sample
            for i in range(0,yagg.shape[0],visthin):
                plt.plot(x,yagg[i,:],'ko',alpha = 0.05)
            ax.set_aspect('equal','datalim')
            add_identity_line(ax)
            # highligh ql and qu
            if v=='q':
                u = model.u
                l = model.l
                for i in range(0,yagg.shape[0],visthin):
                    plt.plot(x[l],yagg[i,l],'go',alpha = 0.05)
                    plt.plot(x[u],yagg[i,u],'ro',alpha = 0.05)

            plt.title(v)
            plt.tight_layout()
            js(fig,'versus_density_'+str(v))

    # TODO: Add titles everywhere!
    # TODO: Save everywhere


    #>>>>> Predictive curves <<<<<
    predictive_curves(q_source="truth")
    #>>>>> HIST w/optional truth <<<<<<

    #>>>>> Corner w truth <<<<<<

    #lnp tracplot

    # plt.show()
    # input('enter')
    #>>>> TEXT SUMMARIES <<<<
    def latex_format(n,u, precision = 1):
        """
        returns estimate with uncertainties in a string format digestible by latex tables
        :param n: number
        :param u: uncertainty
        :return: string to be copypasted into latex tables with some package enabled (siunitx I guess)
        """
        if u<0:
            raise ValueError("uncertainty cannot be negative")
        x = ufloat(n,u)
        s = ("{:."+str(precision)+"uS}").format(x)
        # s = ("{:."+str(precision)+"L}").format(x)
        # This removes a decimal point in uncertainty
        # so that flipping LaTex SiunitX can swallow it
        m = re.search('([^\(]+\()([^\(]+\))', s)
        g1 = m.group(1)
        g2 = m.group(2)
        g2 = re.sub('\.','',g2)
        s = g1+g2
        return s


    def printout(v,yagg,f,x):
        m = yagg.mean(axis = 0)
        h = np.percentile(yagg, 95, axis = 0)
        l = np.percentile(yagg, 5, axis = 0)
        res = m - x
        bias = res.mean()
        rms = np.sqrt(np.mean(np.square(res))) #This is not RMS of residual distribution, but RMS error of estimation!
        (corrcoeff, p) = st.pearsonr(m,x)
        u = 0.5*(h-l)
        e = np.abs(res)/x
        mae = np.abs(res).mean()
        max_abs_err = np.abs(res).max()
        max_rel_err = e.max()

        print(v,file=f)
        print("---------------------------------------------------------",file=f)
        print("{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}".format("mean","(h-l)/2","truth","res","h","l","e","latex"),file=f)
        print("=========================================================",file=f)
        for i in range(len(m)):
            print("{:10.3g},{:10.3g},{:10.3g},{:10.3g},{:10.3g},{:10.3g},{:10.3g},{:>30}".format(
                m[i],u[i],x[i],res[i],h[i],l[i],e[i],latex_format(m[i],u[i],conf.precision)),
                file=f)
        print("---------------------------------------------------------",file=f)
        print("RMS(E_post(x)-x): {:10.3g}".format(rms),file=f)
        print("MAE = E(|E_post(x)-x|): {:10.3g}".format(mae),file=f)
        print("max(|E_post(x)-x|): {:10.3g}".format(max_abs_err),file=f)
        print("max relative error: {:10.3g}".format(max_rel_err),file=f)
        print("Corr coeff: {:10.3g},  p={:10.3g} ".format(corrcoeff, p),file=f)
        print("\n\n",file=f)


    def to_text_file(fname,varlist,formatter):
        with open(fname, 'a') as f:
            for v in varlist:
                yagg = np.vstack((trace[c][v][conf.burn::conf.thin].squeeze()
                                  for c in conf.chains))
                x = truth[v]
                formatter(v,yagg,f,x)

    if conf.text_vars:
        fname = os.path.join(output_dir, 'analysis.txt')
        with open(fname, 'w') as f:
            print('truth source:{}'.format(conf.source),file = f)
            print('',file=f)
        to_text_file(fname,conf.text_vars,printout)
        with open(fname, 'a') as f:
            smagg= np.vstack((trace[c]['S'][conf.burn::conf.thin].squeeze()
                              for c in conf.chains))
            bmagg = dict()
            for k in range(model.K):
                bmagg[k]= np.vstack((trace[c]['b'+str(k)][conf.burn::conf.thin].squeeze()
                           for c in conf.chains))


            #>>>>>>>>>>  Figure of Merit  <<<<<<<<<<
            z = np.linspace(model.ql,model.qu,100)
            deriv = np.ones_like(z)[:,None,None]*bmagg[1]
            for k in range(2,model.K):
                deriv = z[:, None, None] ** (k - 1) * bmagg[k]*k +deriv

            def fm_versions(fmagg, suffix = ""):
                """
                A "temporary" function to process two different variants of Fm calcualtion
                One is to get an honest sample expectation that is noisy but gives realistic CR.
                Another is to try to stabilize the situation by using expected sigma_m in the
                denominator, at the expense of unrealisically narrow CRs.
                The difference is most noticeable with noisy, non-gaussian data.
                :param fmagg: array of calculated Fms , per sample per method
                :param suffix: what to append to filenames and axes
                :return: nothing
                """
                dert = np.ones_like(z)[:,None]*truth['b1'] # dert === derivative true
                for k in range(2,model.K):
                    dert = z[:, None] ** (k - 1) *truth['b'+str(k)]*k +dert
                adert = np.abs(dert)

                ederiv =deriv.mean(axis = 1 ) #expectation of derivative

                def process_figure_of_merit(kind='worst'):
                    if kind=='worst': #looking for minimal true FoM
                        xm = np.argmin(adert,axis=0)
                        ym = np.argmin(np.abs(ederiv),axis = 0)
                    else: #looking for the best FoM
                        kind = ''
                        xm = np.argmax(adert,axis=0)
                        ym = np.argmax(np.abs(ederiv),axis = 0)
                    M = len(xm)
                    x = np.zeros(M)
                    y = np.zeros((fmagg.shape[1],M))
                    for m in range(M):
                        x[m] = adert[xm[m],m]/truth['S'][m]
                        y[:,m] = fmagg[ym[m],:,m]
                    printout('Fm'+kind+''+suffix,y,f,x)
                    plot_verus('Fm'+kind+''+suffix,y,x,suppress_aspect=True)
                    plot_verus('Fm'+kind+''+suffix+' ',y,x,suppress_aspect=False)



                process_figure_of_merit('worst')
                process_figure_of_merit('')

            fmagg = np.abs(deriv/smagg.mean(axis=0))
            fm_versions(fmagg,'_stabilized')
            fmagg = np.abs(deriv/smagg)
            fm_versions(fmagg,'')

            #>>>>>>>>>>  Chebyshev norm (max bias w/o noise) <<<<<<<<<<
            biast = z[:, None]*truth['b0']
            for k in range(1,model.K):
                biast = z[:, None] ** (k) *truth['b'+str(k)] +biast

            biase = z[:, None, None]*bmagg[0]
            for k in range(2,model.K):
                biase = z[:, None, None] ** (k) * bmagg[k] +biase

            biasd = z[:, None, None]*(bmagg[0]-truth['b0'])
            for k in range(2,model.K):
                biasd = z[:, None, None] ** (k) * (bmagg[k]-truth['b'+str(k)])\
                        +biasd
            max_aver_abs_diff = np.abs(biasd.mean(axis = 1)).max(axis = 0)
            print('Max average absolute difference between predictive and true curve',file = f)
            print(max_aver_abs_diff,file=f)
            biasd = np.abs(biasd)#difference
            max_cheb_sample = biasd.max(axis=0)
            M = max_cheb_sample.shape[-1]
            dummy_x = np.array([0.]*M)
            printout('Cheb distance from truth',max_cheb_sample,f,x=dummy_x)
            biase = np.abs(biase-z[:,None,None])
            biast = np.abs(biast-z[:,None])
            where_max = np.argmax(biast,axis=0)
            M = len(where_max)
            x = np.zeros(M)
            y = np.zeros((biase.shape[1],M))
            for m in range(M):
                x[m] = biast[where_max[m],m]
                y[:,m] = biase[where_max[m],:,m]
            printout('Max bias',y,f,x)
            plot_verus('Max abs bias',y,x)





    # TODO: split-trace R-hat
    # TODO: cross-trace R-hat


if __name__=="__main__":
    main()
