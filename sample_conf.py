# force_chain_number = 4
nsamples_per_chain = 3000 #3000 is for testing do more for production:
chains = 16
tune = 1*1000 #600 is just enough with Potential on synthetic data
njobs = 1
discard_tuned_sample = False
approx = False
# Fasle to use normal tuning
# 'advi' for ADVI
# 'fullrank_advi' for FullRankADVI
# 'svgd' for Stein Variational Gradient Descent
# 'asvgd' for Amortized Stein Variational Gradient Descent
# 'nfvi' for Normalizing Flow with default scale-loc flow
# 'nfvi=<formula>' for Normalizing Flow using formula
block_size = 1000
from_truth = False
init = 'advi+adapt_diag'
rescale = True

## >>>>>>>>>> ADVANCED <<<<<<<<<<<<<<
testl = 0. #lower bound of the interval to which x is rescaled
testu = 1. #upper bound of the interval to which x is rescaled
order_sigmoid_scale = 100. # how steep shold the sigmoid that emulates order enforcement must be
nuts_kwargs = {'integrator'}

# TODO: MOVE PREPROCESSING PARAMS TO THIS FILE (and have an option to disable it)
# TODO: SEED ADVI
# TODO: CORNERS
# TODO: PLAY WITH SCALE (TESTVAL MUST BE ADJUSTED SIMULTANEOUSLY)
# TODO: ADD SRS TO SAMPLING LOOP (AND REMOVE FROM POSTPROCESSING)
# TODO: consider dispensing with testvals e.g. by manually coding ordered transform potential (see forum)

# DONE: MORE DATA TO SEE IF SIGMA WOULD STABILIZE : IT DOES
# Samping itself becomes more robust (no divergences occur)
