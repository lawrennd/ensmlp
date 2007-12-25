function [A, lnA] = priorinvcov(net);

% PRIORINVCOV Returns the diagonal of the inverse covariance matrix of the prior
% FORMAT
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP

switch net.alphaprior.type

case 'single'
  A = ones(net.nwts, 1)*net.alpha;
  lnA = ones(net.nwts, 1)*net.lnalpha;
  
otherwise
  tempmat = nrepmat(net.alpha', 1, net.nwts).*net.alphaprior.index;
  tempmat(find(tempmat==0)) = 1;
  A = full(prod(tempmat, 2));
  tempmat = nrepmat(net.lnalpha', 1, net.nwts).*net.alphaprior.index;
  lnA = full(sum(tempmat, 2));
  
end
