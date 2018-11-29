import numpy as np
import os
import pickle
import analyse_conf as conf
import sys
import util


def rangespec_to_list(rangespec):
    """
    parse the range specification
    :param rangespec: string of form "1,3-5,7,8,11-13"
    :return: list with inegers specified by rangespec,e.g. [1,3,4,5,7,8,11,12,13]
    """
    out = set()
    items = rangespec.split(',')
    for i in items:
        if '-' in i:
            a, b = i.split('-')
            out = out.union(set(range(int(a),int(b))))
        else:
            out.add(int(i))
    return sorted(list(out))

def main(chainspec = False, tracedir = 'traces'):
    if chainspec:
        dirlist = list(map(str,rangespec_to_list(chainspec)))
    else: # process all folders
        dirlist = util.getdirlist(tracedir)

    for i in dirlist:
        outdir = os.path.join(tracedir,i,'analysis')
        outfile = os.path.join(outdir,'q.csv')
        os.makedirs(outdir,exist_ok=True)
        ## chains below is a single element list, yes :(
        export_q(chains = [int(i)],tracedir = tracedir,odir = outdir)

def export_q(chains = None, odir = '.', ofname = 'q.csv',tracedir = 'traces'):
    """ Export q expectations from MCMC traces to a csv file"""

    ofname = os.path.join(odir,ofname)
    ## Read traces
    trace = dict() # keys are chain numbers, but not necesserily sequential
    for c in chains:
        fname = os.path.join(tracedir,str(c))
        fname = os.path.join(fname,'trace.pkl')
        with open(fname, 'rb') as f:
            trace[c] = pickle.load(f)[0] # This assumes that there is only one chain per file
    ## Calculate averages
    qagg = np.vstack(
        (trace[c]['q'][conf.burn:conf.end:conf.thin].squeeze() for c in chains)
    )
    q = qagg.mean(axis = 0)
    ## Save as csv
    np.savetxt(ofname , q, delimiter=",")


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv[1])
