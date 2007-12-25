function net = smoothunpak(net, w)

% SMOOTHUNPAK Distribute smoothing parameters in W across the NET structure.
% FORMAT
% DESC takes paked smoothing distribution parameters contained in W and
% distributes it across a NET.
% ARG net : the net structure in which to distribute parameters.
% ARG w : the parameters to distribute.
% RETURN net : the network with the parameters distributed.
%
% SEEALSO : MIXENS, SMOOTHERR, SMOOTHPAK
%            
% BASEDON : Christopher M Bishop and Ian T Nabney, 1996, 1997
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP  

errstring = consist(net, 'smooth');
if ~isempty(errstring);
  error(errstring);
end

if net.npars ~= length(w)
  error('Invalid weight vector length')
end

nin = net.nin;
nhidden = net.nhidden;
nout = net.nout;
nwts = net.nwts;
mark1 = nin*nhidden;
net.w1 = reshape(w(1:mark1), nin, nhidden);
mark2 = mark1 + nhidden;
net.b1 = reshape(w(mark1 + 1: mark2), 1, nhidden);
mark3 = mark2 + nhidden*nout;
net.w2 = reshape(w(mark2 + 1: mark3), nhidden, nout);
mark4 = mark3 + nout;
net.b2 = reshape(w(mark3 + 1: mark4), 1, nout);
mark5 = mark4 + (nin + 1)*nhidden;
%net.d1 = reshape(w(mark4 + 1: mark5), nin + 1, nhidden);
%mark6 = mark5 + (nin + 1)*nhidden*t;
%net.mu1 = reshape(w(mark5 + 1: mark6), nin + 1, nhidden, t);
%mark7 = mark6 + (nhidden + 1)*nout;
%net.d2 = reshape(w(mark6 + 1: mark7), nhidden + 1, nout);
%mark8 = mark7 + (nhidden + 1)*nout*t;
%net.mu2 = reshape(w(mark7 + 1: mark8), nhidden + 1, nout, t);

% Check the structure of the covariance matrix
switch net.covstrct

  case 'diag'
    net.d1 = reshape(w(mark4 + 1: mark5), nin + 1, nhidden);
    mark6 = mark5 + (nhidden + 1)*nout;
    net.d2 = reshape(w(mark5 + 1: mark6), nhidden + 1, nout);

  case {'noded', 'unnoded'}
    t = net.t;
    net.d1 = reshape(w(mark4 + 1: mark5), nin + 1, nhidden);
    mark6 = mark5 + (nhidden + 1)*nout;
    net.d2 = reshape(w(mark5 + 1: mark6), nhidden + 1, nout);
    mark7 = mark6 + (nin + 1)*nhidden*t;
    net.mu1 = reshape(w(mark6 + 1: mark7), nin + 1, nhidden, t);
    %mark7 = mark6 + (nhidden + 1)*nout;
    %net.d2 = reshape(w(mark6 + 1: mark7), nhidden + 1, nout);
    mark8 = mark7 + (nhidden + 1)*nout*t;
    net.mu2 = reshape(w(mark7 + 1: mark8), nhidden + 1, nout, t);
 
  case 'symmetric'
    % Get the matrix U such that Cov = U.U'
    mark6 = mark4 + (nwts+1)*nwts/2;
    net.U = w(mark4+1:mark6)';
end

