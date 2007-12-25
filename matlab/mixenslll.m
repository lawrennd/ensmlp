function lll = mixenslll(net, x, t)

% MIXENSLLL Evaluate lowerbound on likelihood for ensemble learning mixtures.
% FORMAT
% DESC takes a network data structure NET together
% with a matrix X of input vectors and a matrix T of target vectors,
% and evaluates the error function E. The error function is the
% negative of a lowerbound on the log likelihood. It is the negative
% log likelihood plus the Kullback Leibler divergence between the
% Gaussian represented by the network and the true posterior 
% probability of the weights given the data.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN lll : lower bound on log likelihood.
%
% SEEALSO : MIXENS, MIXENSPAK, MIXENSUNPAK, MIXENSFWD, MIXENSGRAD
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1999

% ENSMLP

lll = enslll(net.ens(1), x, t, mixenserr(net, x, t));
