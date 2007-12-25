function [e, E_D, eprior, entropy] = enserr(net, x, t, flag)

% ENSERR Evaluate error function for 2-layer ensemble network.
% FORMAT
% DESC takes a network data structure NET together
% with a matrix X of input vectors and a matrix T of target vectors,
% and evaluates the error function E. The error function is the
% negative of a lowerbound on the log likelihood. It is the negative
% log likelihood plus the Kullback Leibler divergence between the
% Gaussian represented by the network and the true posterior 
% probability of the weights given the data.
% ARG net : the network for which error is required.
% ARG x : the input data.
% ARG t : the target data.
% RETURN e : the error on the data.
%
% DESC also returns the 
% data, prior and entropy components of the total error.
% RETURN E : total error.
% RETURN EDATA : data error.
% RETURN EPRIOR : prior error.
% RETURN ENTROPY : entropy error.
%  
% SEEALSO : ENS, ENSPAK, ENSUNPAK, ENSFWD, ENSGRAD
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence, 1998

% ENSMLP
  
% Check arguments for consistency


% Evaluate the prior contribution to the error.

switch net.covstrct
 case 'none'
  y = ensoutputexpec(net, x);
  E_D = 0.5*sum(sum((y - t).^2));
  if isfield(net, 'beta')
    e1 = net.beta*E_D;
  else
    e1 = E_D;
  end
  if isfield(net, 'alpha')
    w = enspak(net);
    if size(net.alpha) == [1 1]
      eprior = 0.5*(w*w');
      e2 = eprior*net.alpha;
    else
      eprior = 0.5*(w.^2)*net.alphaposterior.index;
      e2 = eprior*net.alpha;
    end
  else
    eprior = 0;
    e2 = 0;
  end
  e = e1 + e2;  
 otherwise
  if nargin == 4
    [e1, E_D] = ensdata_error(net, x, t, 'beta', flag);
  else
    [e1, E_D] = ensdata_error(net, x, t, 'beta');
  end
  eprior = ensprior_error(net);
  entropy = ensentropy_error(net);
  e = e1 + eprior - entropy;  
end
