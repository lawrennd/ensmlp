function lll = enslll(net, x, t, origerr)

% ENSLLL Evaluate lowerbound on likelihood for ensemble learning.
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
% SEEALSO : ENS, ENSPAK, ENSUNPAK, ENSFWD, ENSGRAD
%
% COPYRIGHT Neil D. Lawrence and Mehdi Azzouzi, 1999

% ENSMLP
  
% Check arguments for consistency

errstring = consist(net, 'ens', x, t);
if ~isempty(errstring);
  error(errstring);
end
net = enshypermoments(net);
% Calculate entropy of noise hyperposterior Q(\beta| D)
HQ_beta = sum(gammaentropy(net.betaposterior.a, net.betaposterior.b));
% Calculate entropy of weight hyperposterior Q(\alpha| D)
HQ_alpha = sum(gammaentropy(net.alphaposterior.a, net.alphaposterior.b));
% Calculate expectation of hyperprior with respect to
% hyperposterior
explnPalpha = sum(net.alphaprior.a.*log(net.alphaprior.b) ...
    + (net.alphaprior.a - 1).*net.lnalpha ...
    - net.alphaprior.b.*net.alpha - gammaln(net.alphaprior.a));
explnPbeta = sum(net.betaprior.a.*log(net.betaprior.b) ...
    + (net.betaprior.a - 1).*net.lnbeta ...
    - net.betaprior.b.*net.beta - gammaln(net.betaprior.a));

% Function enserr contains moments of log(P(W, D| \alpha, \beta) and 
% entropy under the distribution Q(W|D, \alpha, \beta)
if nargin < 4
  origerr = enserr(net, x, t, 'hess');
end
origlll = -origerr;

lll = origlll + HQ_beta + HQ_alpha ...
    + explnPalpha + explnPbeta; 




