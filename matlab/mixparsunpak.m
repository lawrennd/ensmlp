function net = mixparsunpak(net, w)

% MIXPARSUNPAK Distribute mixture parameters in W across the NET structure.
% FORMAT
% DESC takes a paked mixture parameters contained in W and
% distributes it across a NET.
% ARG net : the net structure in which to distribute parameters.
% ARG w : the parameters to distribute.
% RETURN net : the network with the parameters distributed.
%
% SEEALSO : MIXENS, MIXPARSERR, MIXPARSPAK
%            
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP  

errstring = consist(net, 'mixpars');
if ~isempty(errstring);
  error(errstring);
end

M = net.M;

% Unpak the mixing and lambda coeff
%net.z = w(1:M);
%net.y = w(M+1:2*M);
%mark1 = 2*M+1;
net.y = w(1:M);
mark1 = M+1;
% For each component unpak the mixture and the smooth distributions
for m = 1:M
  net.smooth(m) = smoothunpak(net.smooth(m), w(mark1:mark1+net.smooth(m).npars-1));
  mark1 = mark1 + net.smooth(m).npars;
end




