"""
This is a config file for the analyse.py script
"""
from model import K

#>>> NAMES <<<
qname = "leftch4p"
#>>> SAMPLE <<<
burn = -1000 # burn samples per chain TODO: rename this to warm-up or tune
thin = 1 #thinning
chains = [0]# [0,1,2,3,4,5,6,7] #  [0,1,2,3] #which chains to use
end = -1

#>>> DEFAULTS <<<
fig_size = (10/2.54,7.5/2.54)
square_fig_size = (7.5/2.54,7.5/2.54)
b_list = ['b'+str(k) for k in range(K)]
bn_list = []# ['bn'+str(k) for k in range(K)]
aux_list= ['a0','a1','order_control']
#>>> TRACEPLOTS <<<
trace_vars = bn_list + \
             b_list + \
             aux_list + \
             ['q','S','R'] #set to a falsy value to skip traceplots
hist_vars = bn_list + \
             b_list + \
             aux_list + \
             ['q','S','R'] #set to a falsy value to skip traceplots
corner_vars = [[]]
#======================================================================
#  GRAPHS FOR VALIDATION: WORK ONLY WHEN TRUTH IS AVAILABLE
#======================================================================
source = 'lse' #'lse'|'gen' what to use as true values of model parameters
#>>> VERSUS PLOTS <<<
versus_vars = b_list+['q','S','R'] #set to a falsy value to skip versus  plots
#>>> VERSUS DENSITY PLOTS <<<
versus_density_vars = versus_vars #set to a falsy value to skip versus  plots
genuine_mm = False
#===============================================================================
#  TEXT SUMMARIES
#===============================================================================
text_vars = versus_vars
precision = 2
