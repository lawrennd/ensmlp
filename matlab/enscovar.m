function [C1, C2, C3, Covar] = enscovar(net)

% ENSCOVAR Combines the parameters mu and d to produce the covariance matrix.
% FORMAT 
% DESC combines the parameters mu and d to produce the covariance matrix.
% ARG net : the network for which covariance is required.
% RETURN Cuu : covariance for first layer of weights.
% RETURN Cvv : covariance for second layer of weights.
% RETURN Cuv : cross covariance between layers.
%	
% SEEALSO : ens, ensunpak, ensfwd, enserr, ensbkp, ensgrad, ensinv
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP
  
% Check arguments for consistency

errstring = consist(net, 'ens');
if ~isempty(errstring);
  error(errstring);
end

nw1 = net.nhidden*net.nin; 		%number of weights in layer 1
nb1 = net.nhidden; 			%number of biases in layer 1
nw2 = net.nhidden*net.nout; 		%number of weights in layer 2
nb2 = net.nout; 			%number of biases in layer 2
np1 = nb1 + nw1; 			%number of parameters in layer 1
np2 = nb2 + nw2; 			%number of parameters in layer 2
switch net.covstrct
  
case 'none'
  C1 = zeros(np1);
  C2 = zeros(np2);
  C3 = zeros(np1, np2);

case 'diag' 				   
  C1 = diag([net.d1(:)'.^2 net.db1.^2]);
  C2 = diag([net.d2(:)'.^2 net.db2.^2]);
  C3 = zeros(np1, np2);
  
case 'noded'     
  mu1 = reshape(net.mu1, nw1, net.t);
  mub1 = reshape(net.mub1, nb1, net.t);
  tmu1 = [mu1; mub1];
  mu1mu1T = tmu1*tmu1';
  C1 = diag([net.d1(:)'.^2 net.db1.^2])+mu1mu1T;
  mu2 = reshape(net.mu2, nw2, net.t);
  mub2 = reshape(net.mub2, nb2, net.t);
  tmu2 = [mu2; mub2];
  mu2mu2T = tmu2*tmu2';
  C2 = diag([net.d2(:)'.^2 net.db2.^2])+mu2mu2T;
  % Template needs fixing
  template = zeros(size(C1));
  for row = 1:net.nhidden
    range = (1+tnin*(row-1)):(tnin*row);
    template(range, range) = 1;
  end
  C1 = C1.*template; 	
  template = zeros(size(C2));
  for row = 1:net.nout
    range = (1+tnhidden*(row-1)):(tnhidden*row);
    template(range, range) = 1;
  end
  C2 = C2.*template; 	
  C3 = zeros(nw1, nw2);
  
case 'layered'
  mu1 = reshape(net.mu1, nw1, net.t);
  mub1 = reshape(net.mub1, nb1, net.t);
  tmu1 = [mu1; mub1];
  mu1mu1T = tmu1*tmu1';
  C1 = diag([net.d1(:)'.^2 net.db1.^2])+mu1mu1T;
  mu2 = reshape(net.mu2, nw2, net.t);
  mub2 = reshape(net.mub2, nb2, net.t);
  tmu2 = [mu2; mub2];
  mu2mu2T = tmu2*tmu2';
  C2 = diag([net.d2(:)'.^2 net.db2.^2])+mu2mu2T;
  C3 = zeros(nw1, nw2);
  
case 'full'
  mu1 = reshape(net.mu1, nw1, net.t);
  mub1 = reshape(net.mub1, nb1, net.t);
  mu2 = reshape(net.mu2, nw2, net.t);
  mub1 = reshape(net.mub2, nb2, net.t);
  mu = [mu1; mub1; mu2; mub2];
  mumuT = mu*mu';
  C = diag([net.d1(:); net.d2(:)].^2) + mumuT;
  C1 = C(1:np1, 1:np1);
  C2 = C(np1+(1:np2), np1+(1:np2));
  C3 = C(1:np1, np1+(1:np2));    
end


% Collect the full covariance matrix
Covar = [C1 C3; C3' C2];





