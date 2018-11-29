# How to use the software 
## Sampling 
Put your measurement data into `_data.csv` file. Open `model.py` to specify
parameters of the priors and the degree of the bias polynomial. Edit
`sample_conf.py` to specify sampler configuration. Run `sample.py` to generate
mcmc samples. 

In the current implementation we sample a single chain, but you can run
several sampling processes (as opposed to threads from pymc3) in parallel.
This design was chosen way because more often than not some of the chains get
stuck in the local maximum of probability and will have to be rejected (see
below).

The chains are saved in pickle format every `block_size` steps. If desired
sampling can be resumed by running sample.py (specify options).

## Analysis

We include analysis scripts that were used to prepare figures for
publication.
