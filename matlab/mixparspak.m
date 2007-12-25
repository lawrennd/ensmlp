function p = mixparspak(net)

% MIXPARSPAK Combines the mixture distribution parameters into one vector.
% FORMAT
% DESC combines the mixture distribution parameters into one vector.
% ARG net : the network to extract parameters from.
% RETURN w : the parameters extracted from the network.
%            
% SEEALSO : mixpars, mixparserr, mixparsgrad
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP
  
% Check arguments for consistency
errstring = consist(net, 'mixpars');
if ~isempty(errstring);
  error(errstring);
end

M = net.M;

% Pak the mixing and lambda coefficients
%/~
%p = [net.z net.y];
%~/
p = [net.y];
% For each component pak the mixture and the smooth distributions
for m = 1:M
  p = [p smoothpak(net.smooth(m))];
end




