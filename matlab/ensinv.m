function [I1, I2, I3] = ensinv(net)

% ENSINV Combines the parameters mu and d for the inverse covariance.
% FORMAT 
% DESC combines the parameters mu and d for the inverse covariance.
% ARG net : the network for which inverse covariance is required.
% RETURN Iuu : inverse covariance for input layer.
% RETURN Ivv : inverse covariance for output layer.
% RETURN Iuv : cross covariance between input and ouptut layer.
%
% SEEALSO : ENS, ENSUNPAK, ENSFWD, ENSERR, ENSBKP, ENSGRAD, ENSCOVAR
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP
  
% Check arguments for consistency
errstring = consist(net, 'ens');
if ~isempty(errstring);
  error(errstring);
end

tnin = net.nin + 1;
tnhidden = net.nhidden + 1;

[Cuu, Cvv, Cuv] = enscovar(net);

nw1 = net.nhidden*(net.nin);
nw2 = net.nhidden*net.nout;
nb1 = net.nhidden;
nb2 = net.nout;
np1 = nw1 + nb1;
np2 = nw2 + nb2;

switch net.covstrct

  case 'none'
    I1 = diag(np1)*inf;
    I2 = diag(np2)*inf;
    I3 = zeros(np1, np2);

case 'diag'
  
  I3 = zeros(np1, np2);
  I1 = diag(ones(1, np1)./([net.d1(:)'.^2 net.db1.^2]));	
  I2 = diag(ones(1, np2)./([net.d2(:)'.^2 net.db2.^2]));	

otherwise
  error('Covariance function not yet implemented.')
  
end







