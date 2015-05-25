ENSMLP software
Version 0.1		Tuesday 25 Dec 2007 at 00:45

This software release resulted from a request for access to the software from the Bayesian Committee tech report. Original software was written across 1998 and 1999. First release was late 2007.


MATLAB Files
------------

Matlab files associated with the toolbox are:

sumdet.m: Computes the determinant of (I+Q*inv(R))
smoothcovinv.m: Combines the parameters mu and d for the inverse covariance.
get_lambda.m: Function to get lambda given the exp parameter z.
mixensunpak.m: Distribute parameters in W across the NET structure.
enserr.m: Evaluate error function for 2-layer ensemble network.
mmigrad.m: Computes the gradient of the MI with respect to the parameters of the net
enshess.m: Evaluate the Hessian matrix for a multi-layer perceptron network.
enspak.m: Takes parameters from structure and places in a vector.
demTecatorMixEnsGroup.m: Learn Tecator with mixture of ensembles and grouped prior.
demTecatorEnsGroup.m: Demonstrate Ensemble with group prior on Tecator data set.
enshyperprior.m: Create gamma priors for hyperparameters in ensmeble learning.
mmi_Rerr.m: Wrapper function for the mixture mutual information bound.
gammaentropy.m: The entropy of a gamma distribution.
mixensrmnode.m: Remove a node from the network.
demTecatorGpRbfArd.m: Demonstrate GP with RBF ARD prior on Tecator data set.
mixhypergradchek.m: Check gradient of hyper parameters.
enshdotv.m: Evaluate the product of the data Hessian with a vector. 
mixenssamp.m: Sample from the posterior distribution
mmi_Runpak.m: Place parameters of smoothing distribution in a network.
mixenserr.m: Evaluate error function for 2-layer mixtures of ensemble network.
ensaddnode.m: Add a node to a ENS structure.
mixensupdatehyperpar.m: Re-estimate parameters of the hyper posteriors.
ensentropy_grad.m: Entropy term's gradient.
init_R.m: initialises the smoothing distributions
ensupdatehyperpar.m: Re-estimate parameters of the hyper posteriors.
mixenspak.m: Takes parameters from structure and places in a vector.
smooth.m: Create the smoothing distribtuions for the mutual information bound.
ensdata_error.m: Error of the data portion.
mixenslll.m: Evaluate lowerbound on likelihood for ensemble learning mixtures.
smoothrmnode.m: Remove a node from the smoothing distribution.
mmi_Rpak.m: Extract parameters of smoothing distributions from network.
mixens.m: Initialises a neural net for ensemble learning with a mixture.
enshypermoments.m: Re-estimate moments of the hyperparameters.
ensentropy_error.m: Entropy term's contribution to the error.
mixparsunpak.m: Distribute mixture parameters in W across the NET structure.
ensunpak.m: Distribute parameters in W across the NET structure.
enscovar.m: Combines the parameters mu and d to produce the covariance matrix.
mixenshypermoments.m: Re-estimate moments of the hyperparameters for the ensemble mixtures.
mixensgrad.m: Evaluate gradient of error function for 2-layer mixture ensemble network.
demTecatorMixEnsNrd.m: Demonstrate Ensemble with NRD on Tecator data set.
ensoutputexpec.m: gives the expectation of the output function and its square.
enslearn.m: Learn an ensemble neural network from data.
ensprior_grad.m: Prior term's gradient.
ensdata_grad.m: Gradient of the data portion.
demTecatorEnsNrd.m: Demonstrate Ensemble with NRD on Tecator data set.
traceQR.m: Computes the trace of (Q*inv(R))
distR.m: Computes the distance with respect to R
smoothunpak.m: Distribute smoothing parameters in W across the NET structure.
evidlearn.m: Learn an evidence procedure neural network from data.
enslll.m: Evaluate lowerbound on likelihood for ensemble learning.
get_pi.m: Function to get pi given the exp parameter z.
mmi.m: The bound on the mutual information term from a mixture of Gaussians.
ensfwd.m: Forward propagation through 2-layer network.
ensgrad.m: Evaluate gradient of error function for 2-layer ensemble network.
ensderiv.m: Evaluate derivatives of network outputs with respect to weights.
priorinvcov.m: Returns the diagonal of the inverse covariance matrix of the prior
mixenslearn.m: Learn a mixture nsemble neural network from data.
mixensmixmstep.m: re-estimate the mixing coefficients of the mixture.
distsum.m: Computes the distance with respect to R
ensprior_error.m: The prior term's portion of the error.
ensrmnode.m: Remove a node from the network.
mixensoutputexpec.m: for each component gives the expectation of the output.
ensexpgrad.m: expectation of the gradient.
mixparspak.m: Combines the mixture distribution parameters into one vector.
mixparserr.m: Portion of error function associated with mixture parameters.
ensinv.m: Combines the parameters mu and d for the inverse covariance.
mmi_Rgrad.m: Wrapper function for the mixture mutual information gradient.
ensmlpToolboxes.m: Load in required toolboxes for ENSMLP.
ens.m: Create a 2-layer feedforward network for ensemble learning
smoothpak.m: Combines the smoothing distribution parameters into one vector.
mixparsgrad.m: Gradient of error function with respect to mixture parameters.
smoothcovar.m: Combines the parameters mu and d to produce the covariance matrix.
