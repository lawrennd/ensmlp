function net = mixenshypermoments(net)

% MIXENSHYPERMOMENTS Re-estimate moments of the hyperparameters for the ensemble mixtures.
% FORMAT
% DESC re-estimates the moments of the hyperparameters ALPHA and BETA under the hyperposterior. 
% ARG net : the input net.
% RETURN net : the net with the moments updated.
%
% SEEALSO : ENSHYPERPRIOR, ENSGRAD, ENSERR, DEMENS1
%
% COPYRIGHT :  Neil D Lawrence and Mehdi Azzouzi, 1999

% ENSMLP

net.ens(1).alpha = net.ens(1).alphaposterior.a./net.ens(1).alphaposterior.b;
net.ens(1).lnalpha = psi(net.ens(1).alphaposterior.a) ...
    - log(net.ens(1).alphaposterior.b);
net.ens(1).beta = net.ens(1).betaposterior.a./net.ens(1).betaposterior.b;
net.ens(1).lnbeta = psi(net.ens(1).betaposterior.a) ...
    - log(net.ens(1).betaposterior.b); 
for m = 2:net.M
  net.ens(m).alpha = net.ens(1).alpha;
  net.ens(m).lnalpha = net.ens(1).lnalpha;
  net.ens(m).beta = net.ens(1).beta;
  net.ens(m).lnbeta = net.ens(1).lnbeta;
end












