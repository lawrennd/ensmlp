function net = mixens(M, nin, nhidden, nout, func, strctQ, strctR,...
                      tQ, tR, prior, beta, rseed, option)
% MIXENS Initialises a neural net for ensemble learning with a mixture.
% FORMAT 
% DESC initialises a neural net for ensemble learning with a mixture of 
% distributions.
% ARG m : number of components in the mixture approximation.
% ARG nin : number of input dimensions.
% ARG nhidden : number of hidden units to use.
% ARG nout : number of output dimensions to use.
% ARG actfn : the activation function on the output units.
% ARG strctQ : covariance structure of the components of the Q
% distribution.
% ARG strctR : covariance structure of the components of the R
% distribution.
% ARG tQ :
% ARG tR :
% ARG alphaprior :
% ARG betaprior :
% ARG rseed : initial random seed.
% ARG opton : either 'y' or 'n' for softmax version of mixing coefficients.b
%        
% SEEALSO : ENS, SMOOTH, MLPRIOR, ENSPAK, ENSUNPAK, ENSFWD, ENSERR, ENSGRAD
%           
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Mehdi Azzouzi and Neil D. Lawrence, 1998, 1999

% ENSMLP

% Initialise the seed
if nargin < 13
  option = 'y';
end
if nargin > 11
  randn('seed', rseed);
  rand('seed', rseed);
end  

if strcmp(option, 'y') == 0 & strcmp(option, 'n') == 0
  error('The sofmax flag should be either ''y'' or ''n''.');
end

% Check the structure of the covariance and just consider 'diag' form 
% for the moment
net.nin = nin;
net.nhidden = nhidden;
net.nout = nout;

if strcmp(strctQ, 'diag') == 1
  net.type = 'mixens';
  net.M = M;
  % Implement softmax version or not for the mixing coeff
  net.soft = option;
  if strcmp(option, 'n') == 1
    net.pi = ones(1,M);
    net.pi = net.pi ./ sum(net.pi);
  else
    % otherwise use a parameterisation
    net.z = randn(1, M);
  end
  
  net.y = rand(1, M);
  % Create M ensemble learning distributions
  for m = 1:M
    net.ens(m) = ens(nin, nhidden, nout, func, strctQ, tQ, prior, beta);
  end
  
  % Create M smooth distributions R
  for m = 1:M
    net.smooth(m) = smooth(nin, nhidden, nout, strctR, tR);
  end
  
else
  error('Covariance structure not yet implemented.\n');
end

net.npars = net.ens(1).npars;
net.nwts = net.ens(1).nwts;


