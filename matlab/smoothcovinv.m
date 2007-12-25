function [I1, I2, I3, Inverse] = smoothcovinv(net)

% SMOOTHCOVINV Combines the parameters mu and d for the inverse covariance.
% FORMAT 
% DESC combines the parameters mu and d for the inverse covariance.
% ARG net : the network for which inverse covariance is required.
% RETURN Iuu : inverse covariance for input layer.
% RETURN Ivv : inverse covariance for output layer.
% RETURN Iuv : cross covariance between input and ouptut layer.
% RETURN I : the full inverse covariance for the smoothing distributions.
%
% SEEALSO : smooth, smoothcovar, mmi, traceQR, mixensmixmstep
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999

% ENSMLP

% Check arguments for consistency
type = {'ens', 'smooth'};
%errstring = consist(net, 'ens');
if sum(strcmp(net.type, type)) == 0
  error('Undefined covariance structure. Exiting.');
end

tnin = net.nin + 1;
tnhidden = net.nhidden + 1;


nw1 = net.nhidden*(tnin);
nw2 = (net.nhidden + 1)*net.nout;

switch net.covstrct
  
 case 'diag'
  
  I3 = zeros(nw1, nw2);
  I2 = zeros(nw2);
  I1 = zeros(nw1);
  I1 = diag(ones(nw1, 1)./net.d1(:).^2);	
  I2 = diag(ones(nw2, 1)./net.d2(:).^2);	
  Inverse = [I1 I3; I3' I2];
  
 case 'symmetric'
  nwts = nw1+nw2;
  void = ones(nwts);
  void = tril(void);
  C = zeros(nwts, nwts);
  C(find(void==1)) = net.U;
  C = C';
  % Use the gauss elimination trick
  Id = eye(size(C,1));
  A = Id/C;
  Inverse = A'*A;
  I1 = Inverse(1:nw1, 1:nw1);
  I2 = Inverse(nw1+(1:nw2), nw1+(1:nw2));
  I3 = Inverse(1:nw1, nw1+(1:nw2));
  
 otherwise
  error('Covariance function not yet implemented')
 
end







