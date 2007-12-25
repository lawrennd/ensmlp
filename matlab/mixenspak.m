function p = mixenspak(net)

% MIXENSPAK Takes parameters from structure and places in a vector.
% FORMAT
% DESC takes a network data structure NET and combines the component weight matrices bias vectors into a single row vector P.
% ARG net : the network to extract parameters from.
% RETURN w : the parameters extracted from the network.
%            
% SEEALSO : mixens, smoothpak, mixenserr, mixensgrad
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999
  
% ENSMLP

% Check arguments for consistency

errstring = consist(net, 'mixens');
if ~isempty(errstring);
  error(errstring);
end

M = net.M;

% Pak the mixing and lambda coefficients
if strcmp(net.soft, 'y') == 1
  p = [net.z net.y];
else
  p = [net.y];
end

% For each component pak the mixture and the smooth distributions
for m = 1:M
  p = [p enspak(net.ens(m)) smoothpak(net.smooth(m))];
end




