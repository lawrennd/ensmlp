function g = ensentropy_grad(net, x)

% ENSENTROPY_GRAD Entropy term's gradient.
% FORMAT 
% DESC returns the gradient of the entropy part.
% ARG net : the network for which gradient is required.
% ARG x : input locations.
% RETURN g : the gradient of the entropy portion.
%
% SEEALSO : ensgrad, ensprior_grad
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP
  
gnet = net;

% Terms that are not involved in the gradient 
gnet.w1 = zeros(size(net.w1));
gnet.b1 = zeros(size(net.b1));
gnet.w2 = zeros(size(net.w2));
gnet.b2 = zeros(size(net.b2));

% Check for type none to avoid inverse of zero matrix
if strcmp(net.covstrct , 'none')
  g = enspak(gnet);
  return
end

[Iuu, Ivv, Iuv] = ensinv(net);
ndata = size(x, 1);


d1 = diag(Iuu)'.*[net.d1(:)' net.db1];
mark1 = net.nin*net.nhidden;
gnet.d1 = reshape(d1(1:mark1), net.nin, net.nhidden);
mark2 = mark1 + net.nhidden;
gnet.db1 = reshape(d1(mark1+1:mark2), 1, net.nhidden);
d2 = diag(Ivv)'.*[net.d2(:)' net.db2];
mark1 = net.nhidden*net.nout;
gnet.d2 = reshape(d2(1:mark1), net.nhidden, net.nout);
mark2 = mark1 + net.nout;
gnet.db2 = reshape(d2(mark1+1:mark2), 1, net.nout);

switch net.covstrct
 case 'diag' 
  % do nothing  
  %/~
  %case 'noded'
  % This needs fixing
  %  gnet.mu1 = zeros(net.nin, net.nhidden, net.t);
  %  gnet.mub1 = zeros(1, net.nhidden, net.t);
  %  net.mu2 = zeros(net.nhidden, net.nout, net.t);
  %  net.mub2 = zeros(1, net.nout, net.t);
  
  %  for row = 1:net.nhidden
  %range = (1+net.nin*(row-1)):(net.nin*row);
  %gnet.mu1(:, row, :) = ...
  %	ctranspose3d(sumn(nrepmat(Iuu(range, range)', 3, net.t) .* ...
  %	nrepmat(net.mu1(:, row, :), 2, tnin), 1));
  %  end
  
  %  for row = 1:net.nout
  %    range = (1+tnhidden*(row-1)):(tnhidden*row);
  %    ghmu2(:, row, :) = ctranspose3d(sumn(nrepmat(Ivv(range, range)', 3, net.t) .* ...
  %	nrepmat(net.mu2(:, row, :), 2, tnhidden), 1));
  
  %  end
  %~/
 otherwise 
  error('Covariance function not yet implemented')
end

g = enspak(gnet);
  


