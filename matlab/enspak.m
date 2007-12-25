function p = enspak(net)

% ENSPAK Takes parameters from structure and places in a vector.
% FORMAT
% DESC takes a network data structure NET and combines the component weight matrices bias vectors into a single row vector P.
% ARG net : the network to extract parameters from.
% RETURN w : the parameters extracted from the network.
%            
% SEEALSO : ens, smoothpak, enserr, ensgrad
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999
  
% ENSMLP

% Check arguments for consistency

errstring = consist(net, 'ens');
if ~isempty(errstring);
  error(errstring);
end

switch  net.covstrct
case 'none'
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2]; 

case 'diag'
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2, ...
	  net.d1(:)', net.db1, net.d2(:)', net.db2]; 
case 'noded'
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2, ...
	  net.d1(:)', net.db1, net.d2(:)', net.db2,  ...
	  net.mu1(:)', net.mub1, net.mu2(:)', net.mub2];
otherwise
  error('Covariance structure not yet implemented.');
end





