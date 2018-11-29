# model knobs
# polynomial degree+1 #this knob is broken - MUST be 2 for linear model
K = 2

#following 4 params must be float
ql= 0.28 #lower bound on q
qu = 0.48 #upper bound on q
# tolerances are positive!, they are absolute values
el = 0.02 #minimum point tolerance ($\underline\epsilon$)
eu = 0.02 #maximum point tolerance ($\overline\epsilon)

# Taylor expansion origin (point of least predictive error)
q0 = (qu+ql)/2.

# indices of elements for 1-bit prior
# x_lt must be known to be less thatn x_ut
l = 1
u = 34


minimal_xu_xl_distance = 0
