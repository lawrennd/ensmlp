function net = ensunpak(net, w)

% ENSUNPAK Distribute parameters in W across the NET structure.
% FORMAT
% DESC takes a paked ensemble learning structure contained in W and
% distributes it across a NET.
% ARG net : the net structure in which to distribute parameters.
% ARG w : the parameters to distribute.
% RETURN net : the network with the parameters distributed.
%
% SEEALSO : ENS, ENSFWD, ENSERR, ENSBKP, ENSGRAD
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

switch net.covstrct
case 'none'
  if net.nwts ~= length(w)
    error('Invalid weight vector length')
  end
otherwise
  if net.npars ~= length(w)
    error('Invalid weight vector length')
  end
end
nin = net.nin;
nhidden = net.nhidden;
nout = net.nout;

mark1 = nin*nhidden;
net.w1 = reshape(w(1:mark1), nin, nhidden);
mark2 = mark1 + nhidden;
net.b1 = reshape(w(mark1 + 1: mark2), 1, nhidden);
mark3 = mark2 + nhidden*nout;
net.w2 = reshape(w(mark2 + 1: mark3), nhidden, nout);
mark4 = mark3 + nout;
net.b2 = reshape(w(mark3 + 1: mark4), 1, nout);

% Check the structure of the covariance matrix
switch net.covstrct

case 'none'
  % Do nothing
  
otherwise
  mark5 = mark4 + nin*nhidden;
  net.d1 = reshape(w(mark4 + 1: mark5), nin, nhidden);
  mark6 = mark5 + nhidden;
  net.db1 = reshape(w(mark5 + 1: mark6), 1, nhidden);
  mark7 = mark6 + nhidden*nout;
  net.d2 = reshape(w(mark6 + 1: mark7), nhidden, nout);
  mark8 = mark7 + nout;
  net.db2 = reshape(w(mark7 + 1: mark8), 1, nout);

  switch net.covstrct
  case 'diag'
    % Do nothing
    
  case 'noded'
    t = net.t;
    mark9 = mark8 + nin*nhidden*t;
    net.mu1 = reshape(w(mark8 + 1: mark9), nin, nhidden, t);
    mark10 = mark9 + nhidden*t;
    net.mub1 = reshape(w(mark9 + 1: mark10), 1, nhidden, t);
    mark11 = mark9 + nhidden*nout*t;
    net.mu2 = reshape(w(mark10 + 1: mark11), nhidden, nout, t);
    mark12 = mark11 + nout*t;
    net.mub2 = reshape(w(mark11 + 1: mark12), 1, nout, t);

  otherwise
    error('Covariance structure not yet available')
  end
end


