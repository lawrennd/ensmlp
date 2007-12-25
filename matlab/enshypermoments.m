function net = enshypermoments(net)

% ENSHYPERMOMENTS Re-estimate moments of the hyperparameters.
% FORMAT
% DESC re-estimates the moments of the hyperparameters ALPHA and BETA under the hyperposterior. 
% ARG net : the input net.
% RETURN net : the net with the moments updated.
%
% SEEALSO : ENSHYPERPRIOR, ENSGRAD, ENSERR, DEMENS1
%
% COPYRIGHT :  Neil D Lawrence and Mehdi Azzouzi, 1999

% ENSMLP

net.alpha = net.alphaposterior.a./net.alphaposterior.b;
net.lnalpha = digamma(net.alphaposterior.a) - log(net.alphaposterior.b);
net.beta = net.betaposterior.a./net.betaposterior.b;
net.lnbeta = digamma(net.betaposterior.a) - log(net.betaposterior.b); 













