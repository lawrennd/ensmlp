function [h, dh] = enshess(net, x, t, dh)

% ENSHESS Evaluate the Hessian matrix for a multi-layer perceptron network.
% FORMAT
% DESC takes an MLP network data structure NET, a
% matrix X of input values, and a matrix T of target values and returns
% the full Hessian matrix H corresponding to the second derivatives of
% the negative log posterior distribution, evaluated for the current
% weight and bias values as defined by NET.
% ARG net : the network for which it is required.
% ARG x : input locations.
% ARG t : target locations.
% RETURN H : the Hessian.
%
% DESC returns both the Hessian matrix H and the contribution DH arising
% from the data dependent term in the Hessian.
% ARG net : the network for which it is required.
% ARG x : input locations.
% ARG t : target locations.
% RETURN H : the Hessian.
% RETURN dH : the data dependent portion of the Hessian.
%
% DESC takes a network data structure NET, a
% matrix X of input values, and a matrix T of  target values, together
% with the contribution DH arising from the data dependent term in the
% Hessian, and returns the full Hessian matrix H corresponding to the
% second derivatives of the negative log posterior distribution. This
% version saves computation time if DH has already been evaluated for
% the current weight and bias values.
% ARG net : the network for which it is required.
% ARG x : input locations.
% ARG t : target locations.
% ARG dh : vector with which Hessian product is required.
% RETURN H : the Hessian.
% RETURN dH : the data dependent portion of the Hessian.
%
% SEEALSO : MLP, HESSCHEK, MLPHDOTV, EVIDENCE
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

if nargin == 3
  % Data term in Hessian needs to be computed
  dh = datahess(net, x, t);
end
% This line needs generalising to multiple outputs
if isfield(net, 'beta')
  h = net.beta*dh;
else
  h = dh;
end
% Account for the prior
h = h + diag(priorinvcov(net));

function dh = datahess(net, x, t)

dh = zeros(net.nwts, net.nwts);

for v = eye(net.nwts);
  dh(find(v),:) = enshdotv(net, x, t, v);
end

return


