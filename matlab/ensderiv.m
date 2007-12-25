function g = ensderiv(net, x)
% ENSDERIV Evaluate derivatives of network outputs with respect to weights.
% FORMAT
% DESC takes a network data structure NET and a matrix
% of input vectors X and returns a three-index matrix G whose I, J, K
% element contains the derivative of network output K with respect to
% weight or bias parameter J for input pattern I. The ordering of the
% weight and bias parameters is defined by ENSUNPAK.
% ARG net : network for which derivatives is required.
% ARG x : input locations for which derivatives are required.
%
% SEEALSO : ens, enspak, ensgrad
%
% COPYRIGHT : Neil D. Lawrence, 1998
% 
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996

% ENSMLP

% Check arguments for consistency
errstring = consist(net, 'ens', x);
if ~isempty(errstring);
  error(errstring);
end

if ~strcmp(net.actfn, 'linear')
  error('Function not implemented except for linear outputs')
end

ndata = size(x, 1);

g = zeros(ndata, net.nwts, net.nout);

[Cuu, Cvv, Cuv] = enscovar(net);
[Iuu, Ivv, Iuv] = ensinv(net);
ndata = size(x, 1);
tnin = 1 + net.nin;
tnhidden = 1 + net.nhidden;



[qw1, qb1, qw2, qb2, ...
	sw1, sb1, sw2, sb2] = element_grad(net, x);


if isfield(net, 'beta')
  vbeta = net.beta;
else
  vbeta = 1;
end
gw1 = .5*(qw1 - 2*sw1);
gw1 = permute(gw1, [1 4 3 2]);
gb1 = 0.5*(qb1 - 2*sb1);
tgw2 = 0.5*(qw2 - 2*sw2);
tgb2 = 0.5*(qb2 - 2*sb2);

gw2 = zeros(net.nhidden, net.nout, ndata, net.nout);
gb2 = zeros(1, net.nout, ndata, net.nout);

for n = 1:net.nout
  gw2(:, n, :, :) = permute(tgw2, [1 4 3 2]);
  gb2(:, n, :, :) = permute(tgb2, [1 4 3 2]);
end

sw2 = permute(reshape(sw2, [net.nhidden*net.nout, ndata, net.nout]), [3 2 1]);
sb2 = permute(shiftdim(sb2, 1), [3 2 1]);
sw1 = permute(reshape(sw1, [net.nin*net.nhidden, ndata, net.nout]), [3 2 1]);
sb1 = permute(shiftdim(sb1, 1), [3 2 1]);

mark1 = net.nin*net.nhidden;
g(:, 1:mark1, :) = sw1;
mark2 = mark1 + net.nhidden;
g(:, mark1 + 1:mark2, :) = sb1;
mark3 = mark2 + net.nhidden*net.nout;
g(:, mark2 + 1:mark3, :) = sw2;
mark4 = mark3 + net.nout;
g(:, mark3 + 1:mark4, :) = sb2;

function [qw1, qb1, qw2, qb2, ...
	sw1, sb1, sw2, sb2] = element_grad(net, x)

fact = sqrt(2)/2;
w = enspak(net);
[Cuu, Cvv, Cuv] = enscovar(net);
V = [net.w2; net.b2];
tnhidden = net.nhidden + 1; % include the bias
tnin = net.nin + 1;
ndata = size(x, 1);
xTu = x*net.w1 + nrepmat(net.b1, 1, ndata);

Theta = zeros(tnhidden, tnhidden, net.nout);
Thetag = zeros(ndata, tnhidden, net.nout);
vmat = zeros(ndata, net.nhidden, net.nout);
Phi = zeros(ndata, net.nhidden);
g = zeros(ndata, net.nhidden);
gdash = zeros(ndata, net.nhidden);
aTa = zeros(ndata, 1);
store = zeros(ndata, tnin);
a0 = zeros(ndata, 1);

sw2 = zeros(net.nhidden, net.nout, ndata);
sb2 = zeros(1, net.nout, ndata);

sw1 = zeros(net.nin, net.nhidden, ndata, net.nout);
sb1 = zeros(1, net.nhidden, ndata, net.nout);

qw2 = zeros(net.nhidden, net.nout, ndata);
qb2 = zeros(1, net.nout, ndata);

qw1 = zeros(net.nin, net.nhidden, ndata, net.nout);
qb1 = zeros(1, net.nhidden, ndata, net.nout);

for n = 1:net.nout
  rowrange = (1+tnhidden*(n-1)):(tnhidden*n);
  Theta(:, :, n) = [V(:, n)*V(:, n)'];
end

for row =1:net.nhidden
  rowrangei = (1+tnin*(row-1)):(tnin*row - 1);
  rowrangej = [rowrangei (tnin*row)];
  a0(:, :) = xTu(:, row);
  g(:, row) = erf(fact*a0);
  gdash(:, row) = sqrt(2/pi)*exp(-(a0.^2));
end

qgnet=net;
sgnet=net;

xsquare = x.*x;
%We don't want to include the bias at every data point thus 1:net.nhidden
for n = 1:net.nout
  Thetag(:, :, n) = g * Theta(1:net.nhidden, :, n) + ...
      nrepmat(Theta(tnhidden, :, n), 1, ndata);  
  vmat(:, :, n) = nrepmat(net.w2(:, n)', 1, ndata);
end

xsquaremat = nrepmat(permute(xsquare, [2 3 1]), 2, net.nhidden); 
xmat = nrepmat(permute(x, [2 3 1]), 2, net.nhidden); 

% Data points are laid out in dimension 1;
%%%%%%%%%%%%%%%%
gTv = (g*net.w2)+ nrepmat(net.b2, 1, ndata);

sgV = permute([g ones(ndata, 1)], [2 3 1]);
sw2(:, :, :) = nrepmat(sgV(1:net.nhidden, :, :), 2, net.nout);
sb2(:, :, :) = nrepmat(sgV(tnhidden, :, :), 2, net.nout);

kbasecomp1 = gdash;
for n = 1:net.nout
  kcomp = vmat(:, :, n).*kbasecomp1;
  kcompmat = matrisize(kcomp, net.nin);
  sw1(:, :, :, n) = kcompmat.*xmat;
  sb1(:, :, :, n) = permute(kcomp, [2 3 1 4]);
end  

qw2 = sw2.*sw2;
qw1 = sw1.*sw1;
qb1 = sb1.*sb1;
qb2 = sb2.*sb2;

function x = matrisize(y, nrep)
x = nrepmat(permute(y, [3 2 1]), 1, nrep);










