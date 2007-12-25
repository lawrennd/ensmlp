function [g, gprior, gdata, gentropy] = ensgrad(net, x, t)

% ENSGRAD Evaluate gradient of error function for 2-layer ensemble network.
% FORMAT
% DESC takes a network data structure NET together with a matrix X of input
% vectors and a matrix T of target vectors, and evaluates the gradient G of
% the error function with respect to the network parameters. The error
% function is the negative of a lowerbound on the log likelihood. It is the
% negative log likelihood plus the Kullback Leibler divergence between the
% Gaussian represented by the network and the true posterior probability of
% the weights given the data.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN g : gradient of lower bound on log likelihood.
%
% DESC also returns separately the data, entropy and prior contributions to
% the gradient. In the case of multiple groups in the prior, GPRIOR is a
% matrix with a row for each group and a column for each weight parameter.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN g : gradient of lower bound on log likelihood.
% RETURN gprior : gradient of the prior portion.
% RETURN gdata : gradient of the data portion.
% RETURN gentropy : gradient of the entropy portion.
%
% SEEALSO : ENS, ENSPAK, ENSUNPAK, ENSSFWD, ENSERR
%
% COPYRIGHT : Neil D Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP
  
% Check arguments for consistency
errstring = consist(net, 'ens', x, t);
if ~isempty(errstring);
  error(errstring);
end

switch net.covstrct
 case 'none'
  gdata = ensdata_grad(net, x, t);
  gprior = ensprior_grad(net);
  
  g = gdata + gprior;
 otherwise
  gentropy = ensentropy_grad(net, x);
  gdata = ensdata_grad(net, x, t);
  gprior = ensprior_grad(net);
  
  g = gdata + gprior - gentropy;
end



























