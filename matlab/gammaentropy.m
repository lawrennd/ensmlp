function H=gammaentropy(a, b)

% GAMMAENTROPY The entropy of a gamma distribution.
%
  
% ENSMLP
  
H = -((a-1).*digamma(a) + log(b) - a - gammaln(a));
