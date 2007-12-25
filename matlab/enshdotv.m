function hdv = enshdotv(net, x, t, v)

% ENSHDOTV Evaluate the product of the data Hessian with a vector. 
% FORMAT
% DESC takes an ENS network data structure NET,
% together with the matrix X of input vectors, the matrix T of target
% vectors and an arbitrary row vector V whose length equals the number
% of parameters in the network, and returns the product of the data-
% dependent contribution to the Hessian matrix with V. The
% implementation is based on the R-propagation algorithm of
% Pearlmutter.
% ARG net : the network for which it is required.
% ARG x : input locations.
% ARG t : target locations.
% ARG v : vector with which Hessian product is required.
% RETURN hdv : the product of the Hessian and the vector.
%
% SEEALSO : ens, enshess, mlp, mlphess, hesschek
%
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence, 1998

% ENSMLP

% Check arguments for consistency
errstring = consist(net, 'ens', x, t);
if ~isempty(errstring);
  error(errstring);
end
if net.covstrct~='none'
  error('Function only valid for no covariance structure')
end
ndata = size(x, 1);

[y, z] = ensfwd(net, x);		% Standard forward propagation.
za = x*net.w1 + ones(ndata, 1)*net.b1;
zprime = sqrt(2/pi)*exp(-.5*za.^2);      % Hidden unit first derivatives.
zpprime = -za.*zprime;      % Hidden unit second derivatives.
vnet = ensunpak(net, v);	% 		Unpack the v vector.

% Do the R-forward propagation.

ra1 = x*vnet.w1 + ones(ndata, 1)*vnet.b1;
rz = zprime.*ra1;
ra2 = rz*net.w2 + z*vnet.w2 + ones(ndata, 1)*vnet.b2;

switch net.actfn

  case 'linear'        %Linear outputs

    ry = ra2;

  otherwise
    error('enshdotv is implemented only for linear activation function')

end

% Evaluate delta for the output units.

delout = y - t;

% Do the standard backpropagation.

delhid = zprime.*(delout*net.w2');

% Now do the R-backpropagation.

rdelhid = zpprime.*ra1.*(delout*net.w2') + zprime.*(delout*vnet.w2') + ...
          zprime.*(ry*net.w2');

% Finally, evaluate the components of hdv and then merge into long vector.

hw1 = x'*rdelhid;
hb1 = sum(rdelhid, 1);
hw2 = z'*ry + rz'*delout;
hb2 = sum(ry, 1);

hdv = [hw1(:)', hb1, hw2(:)', hb2];


