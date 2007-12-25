function [g, gprior, gdata, gentropy] = mixensgrad(net, x, t)

% MIXENSGRAD Evaluate gradient of error function for 2-layer mixture ensemble network.
% FORMAT
% DESC takes a network data structure NET together with a matrix X of input
% vectors and a matrix T of target vectors, and evaluates the gradient G of
% the error function with respect to the network parameters. The error
% function is the negative of a lowerbound on the log likelihood. It is the
% negative log likelihood plus the Kullback Leibler divergence between the
% Gaussian represented by the network and the true posterior probability of
% the weights given the data.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN g : gradient of lower bound on log likelihood.
%
% DESC also returns separately the data, entropy and prior contributions to
% the gradient. In the case of multiple groups in the prior, GPRIOR is a
% matrix with a row for each group and a column for each weight parameter.
% ARG net : network for which likelihood is required.
% ARG x : input data.
% ARG t : target data.
% RETURN g : gradient of lower bound on log likelihood.
% RETURN gprior : gradient of the prior portion.
% RETURN gdata : gradient of the data portion.
% RETURN gentropy : gradient of the entropy portion.
%
% SEEALSO : mixens, mixenspak, mixensupak, mixensfwd, mixenserr
%
% COPYRIGHT : Neil D Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP

% Check arguments for consistency
errstring = consist(net, 'mixens', x, t);
if ~isempty(errstring);
  error(errstring);
end

gcomps = [];

% Check the type of softmax
if strcmp(net.soft, 'y') == 1
  mixing_coeff = get_pi(net.z);
  gmixing_coeff = zeros(1, net.M);
  gz = zeros(1, net.M);
  grad_mix_z = diag(mixing_coeff) - mixing_coeff'*mixing_coeff;
  gz = zeros(1, net.M);

  % Now compute the gradient with respect to z
  for i = 1:net.M
    for j = 1:net.M
      gz(i) = gz(i) + gmixing_coeff(j)*grad_mix_z(j, i);
    end
  end
  
else
  mixing_coeff = net.pi;
end

for m = 1:net.M
  gentropy = ensentropy_grad(net.ens(m), x);
  gdata = ensdata_grad(net.ens(m), x, t);
  gprior = ensprior_grad(net.ens(m));
  % gmixing_coeff(m) = enserr(net.ens(m), x, t);
  gcomps = [gcomps mixing_coeff(m)*(gdata - gentropy + gprior) ... 
      zeros(1, net.smooth(m).npars)];
end

if strcmp(net.soft, 'y') == 1
  % Take the gradient wrt net.z
  gcomps = [gz zeros(1, net.M) gcomps];
else
  % No gradient wrt to net.z
  gcomps = [zeros(1, net.M) gcomps];
end

% Gradients with respect to the lower bound on the mutual information
glmi = mmigrad(net);

g = gcomps - glmi;
if ~isreal(g)
  warning('Gradient is complex')
end

% Don't need the following line anymore as the softmax flag has been implemented
%g = g(net.M+1:size(g, 2));


function g = entropy_grad(net, x)

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
otherwise 
  error('Covariance function not yet implemented')
end

g = enspak(gnet);
  


function x = matrisize(y, nrep)
x = nrepmat(permute(y, [3 2 1]), 1, nrep);

function g = prior_grad(net)

if isfield(net, 'alpha')
  w = enspak(net);
  A = priorinvcov(net);    
  
  switch net.covstrct    
  case 'none'
    g = [A'.*w(1:net.nwts)];
    
  case 'diag'       
    g = [A'.*w(1:net.nwts), A'.*w(net.nwts+(1:net.nwts))];
    
  case 'noded'
    error('Code for noded not yet implemented')
      %g = net.alpha* ...
      %[w(1:ndet.nwts) 2*w((net.nwts+1):net.npars)];
  
  case 'layered'      
    error('Code for layered not yet implemented')
    
  case 'full'
      error('Code for full not yet implemented')
	
  end 
else
  g = 0;
end

















