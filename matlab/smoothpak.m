function p = smoothpak(net)

% SMOOTHPAK Combines the smoothing distribution parameters into one vector.
% FORMAT
% DESC combines the smoothing distribution parameters into one vector.
% ARG net : the network to extract parameters from.
% RETURN w : the parameters extracted from the network.
%            
% SEEALSO : smooth, smootherr, smoothgrad
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP
  
% Check arguments for consistency
errstring = consist(net, 'smooth');
if ~isempty(errstring);
  error(errstring);
end

switch  net.covstrct
case 'none'
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2]; 

case 'diag'
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2, ...
	  net.d1(:)', net.d2(:)']; 
  
case {'noded', 'unnoded'}
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2, ...
	  net.d1(:)', net.d2(:)',  ...
	  net.mu1(:)', net.mu2(:)'];

case 'symmetric'
  p = [net.w1(:)', net.b1, net.w2(:)', net.b2, ...
	  net.U(:)'];
otherwise
  error('Covariance structure not yet implemented.');
end




