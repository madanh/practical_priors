import numpy as np
import pandas as pd
import matplotlib as mpl
import os
# mpl.use('TkAgg')
import  matplotlib.pyplot as plt
fname = 'nonans.csv'
outdir = 'fit_images'+fname

def get_fm(yfit,x,y):
    res = y - np.polyval(yfit,x)
    s = np.std(res)
    return s/yfit[-2]

def plot_fit(ax,x,y,deg,linespec='k-'):
    yfit = np.polyfit(x,y,deg)
    fm =  get_fm(yfit,x,y)
    xlim = ax.get_xlim()
    xrange = np.linspace(xlim[0],xlim[1],100)
    yfit = np.polyval(yfit,xrange)
    ax.plot(xrange,yfit,linespec)
    if deg == 2:
        ax.set_title("F_m(quad) = {:.2f}".format(fm))
    elif deg == 1:
        ax.set_xlabel("F_m(lin) = {:.2f}".format(fm))


d = pd.read_csv(fname)#,delim_whitespace=True)# d for data
try:
    if d.isna().any().any():
        print("WARNING: Missing values present")
        miss = np.where(d.isna())[0]
        for i in miss:
            print("Row {} : \n{}".format(i,d.ix[i]))
except AttributeError:
    print("In this pandas version there is no 'isna' function, but we push on")

ncols = len(d.columns)

os.makedirs(outdir,exist_ok = True)

def calc_geom(ncols,b=1):
    """recursively find the best ratio (closest to 3:4"""
    def best(ncols,b): #lispy
        return np.abs((ncols//(b+1))/(b+1)-4/3)>=((ncols//b)/b-4/3)
    if best (ncols,b): #term cond
        return (ncols,b)
    else:
        return calc_geom(ncols,b+1)
_,b = calc_geom(ncols-1)
a = (ncols-1)//b+1
# iterate over columns for x
for x in d:
    fig, ax = plt.subplots(b,a,figsize=(40,30))
    ax = ax.flatten()
    i = 0
    for y in d:
        if y==x:
            continue
        ax[i].scatter(d[x],d[y])
        plot_fit(ax[i],d[x],d[y],2,'r-')
        plot_fit(ax[i],d[x],d[y],1,'c-')
        ax[i].set_ylabel(y)
        i += 1
    fig.suptitle(x)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir,'{}'.format(x)))
    plt.close(fig)

pass
