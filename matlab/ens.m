function net = ens(nin, nhidden, nout, func, strct, t, alphaprior, betaprior)
% ENS Create a 2-layer feedforward network for ensemble learning
% FORMAT
% DESC takes the number of inputs, hidden units and output units for a
% 2-layer feed-forward network, together with a string FUNC which specifies
% the output unit activation function, and returns a data structure NET. The
% weights are drawn from a zero mean, unit variance isotropic Gaussian, with
% varianced scaled by the fan-in of the hidden or output units as
% appropriate. This makes use of the Matlab function RANDN and so the seed
% for the random weight initialization can be set using RANDN('STATE', S)
% where S is the seed value.  The hidden units use the ERF activation
% function.
%
% The fields in NET are
%	  type = 'ens'
%	  nin = number of inputs
%	  nhidden = number of hidden units
%	  nout = number of outputs
%	  nwts = total number of weights and biases
%	  npars = total number of parameters defining the Gaussian
%         actfn = string describing the output unit activation function:
%	      'linear'
%	      'sigmoid'
%	      'softmax'
%         strct = string describing the structure of the covariance matrix:
%	      ''
%	      ''
%	      ''
%	  w1 = first-layer weight matrix
%	  b1 = first-layer bias vector
%	  w2 = second-layer weight matrix
%	  b2 = second-layer bias vector
%	  d1, d2, mu1 and mu2 parameterise the covariance function.
%	  
% Here W1 has dimensions NIN times NHIDDEN, B1 has dimensions 1 times
% NHIDDEN, W2 has dimensions NHIDDEN times NOUT, B2 has dimensions
% 1 times NOUT. T adjusts the parameterisation of the covariance 
% matrix of the gaussian. The larger T, the greater the complexity 
% and therefore higher accuracy in computation. 
%
% ARG nin : number of inputs. 
% ARG nhidden : number of hidden nodes.
% ARG nout : number of output nodes.
% ARG actfunc : activation function.
% ARG strct : covariance structure, either 'full', 'layered', 'noded', 'diag' or 'none'.
% ARG t : 
% ARG alphaprior :
% ARG betaprior :
%
% SEEALSO : mlpprior, enspak, ensunpak, ensfwd, enserr, ensgrad
%
% COPYRIGHT :  Neil D Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP
  
net.type = 'ens';
net.nin = nin;
net.nhidden = nhidden;
net.nout = nout;
net.nwts = (nin + 1)*nhidden + (nhidden + 1)*nout;
actfns = {'linear', 'logistic', 'softmax'};
covstrcts = {'full', 'layered', 'noded', 'diag', 'none'};


if sum(strcmp(func, actfns)) == 0
  error('Undefined activation function. Exiting.');
else
  net.actfn = func;
end

% Check the consistency between 'diag' and t
if strcmp(strct, 'diag') == 1 & t ~= 0
  error('Cannot deal with diagonal matrix and t ~= 0.\n');
end

% Check the covariance structure
if sum(strcmp(strct, covstrcts)) == 0
  error('Undefined covariance structure. Exiting.');
else 
  net.covstrct = strct;
end

% Collect the number of parameters
if strcmp(net.covstrct, 'none')
  net.npars = net.nwts;
else
  net.npars = net.nwts + (nin + 1)*nhidden*(t + 1) + ...
      (nhidden+ 1)*nout*(t + 1); 
end

% Check the prior on alpha
if isstruct(alphaprior)
  net.alphaprior = alphaprior;
  net.alphaposterior = alphaprior;
  if strcmp(net.covstrct, 'none')
    net.index = alphaprior.index; 
  end
else
  error('alphaprior must be a structure');
end  

% Check for the presence of a prior on beta
if isstruct(betaprior)
  net.betaprior = betaprior;
  net.betaposterior = betaprior;
else
  error('betaprior must be a structure');
end
% Set up the moments of alpha and beta 
net = enshypermoments(net);

% Initialise the mean and covariance of the weights and biases
switch net.covstrct
case 'none'
  % Give weights some value for symmetry breaking
  initw = randn(net.nwts, 1)/100;
  net = ensunpak(net, initw);

otherwise
  % Give weights some value for symmetry breaking - shouldn't really be done
  initw = randn(net.nwts, 1)/100;
  %initw = zeros(net.nwts, 1);
  invdiagC = priorinvcov(net);
  diagC = 1./invdiagC;
  C = diag(diagC);
  initd = ones(net.nwts, 1)/10;% sqrt(diagC);
  
  switch net.covstrct
  case 'diag'
    init = [initw; initd];
    net = ensunpak(net, init);
    
  otherwise
    error('Covariance structure not yet fully implemented')
  
  end
end




