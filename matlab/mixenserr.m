function [e, E_D, eprior, entropy] = mixenserr(net, x, t)

% MIXENSERR Evaluate error function for 2-layer mixtures of ensemble network.
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
% SEEALSO : MIXENS, MIXENSPAK, MIXENSUNPAK, MIXENSOUTPUTEXPEC, MIXENSGRAD
%
% BASEDON : Christopher M. Bishop and Ian T. Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP 
  
% Check arguments for consistency

% p = y;
%net = mixensunpak(net, p);

errstring = consist(net, 'mixens', x, t);
if ~isempty(errstring);
  error(errstring);
end

% Get the mixing coeff depending on the parametrisation
if strcmp(net.soft, 'y') == 1
  mixing_coeff = get_pi(net.z);
else
  mixing_coeff = net.pi;
end

% Calculate the portion of the entropy due to each component
entropy = 0;
for m = 1:net.M
  entropy = entropy + ...
      mixing_coeff(m)*ensentropy_error(net.ens(m));
end

% Calculate the portion of the data contribution due to each component
e1 = 0;
e2 = 0;
for m = 1:net.M
  e1 = e1 + ensdata_error(net.ens(m), x, t, 'beta')*mixing_coeff(m);
  % Evaluate the prior contribution to the error.
  if isfield(net.ens(m), 'alphaprior')
    e2 = e2 + ensprior_error(net.ens(m))*mixing_coeff(m);
  else
    e2 = 0;
  end
end

% Evaluate the mutual information contribution and remove from entropy.
entropy = entropy  +  mmi(net);

% disp([e1 e2 entropy])
e = e1 + e2 - entropy;  
if ~isreal(e)
  warning('Error is not real')
end

 

