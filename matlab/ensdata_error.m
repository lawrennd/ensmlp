function [e, E_D] = ensdata_error(net, x, t, betaflag, flag)

% ENSDATA_ERROR Error of the data portion.
% FORMAT
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1999

% ENSMLP
  
if nargin<4
  betaflag = 'beta';
end
errstring = consist(net, 'ens', x, t);
if ~isempty(errstring);
  error(errstring);
end

if nargin == 5
  [sexp, qexp] = ensoutputexpec(net, x, flag, x, t);
else
  [sexp, qexp] = ensoutputexpec(net, x);
  
end

ndata = size(x, 1);
temp = sum(qexp -2*t.*sexp + t.*t, 1);
E_D = 0.5*(sum(temp, 2));


if isfield(net, 'betaposterior') & strcmp(betaflag, 'beta')
  e = .5*(sum(net.beta.*temp) - ndata*sumn(net.lnbeta, 1) ...
      + net.nout*ndata*log(2*pi));
else
  e = E_D;
end
