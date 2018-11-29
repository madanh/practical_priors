# model knobs
# polynomial degree+1 #this knob is broken - MUST be 2 for linear model
K = 2

#following 4 params must be float
ql= 0.26 #lower bound on q
qu = 0.42 #upper bound on q
# tolerances are positive!, they are absolute values
el = 0.04 #minimum point tolerance ($\underline\epsilon$)
eu = 0.02 #maximum point tolerance ($\overline\epsilon)

# Taylor expansion origin (point of least predictive error)
q0 = (qu+ql)/2.

# indices of elements for 1-bit prior
# x_lt must be known to be less thatn x_ut
l = 3
u = 26


minimal_xu_xl_distance = 0
