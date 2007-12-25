function [C1, C2, C3, Covar] = smoothcovar(net)

% SMOOTHCOVAR Combines the parameters mu and d to produce the covariance matrix.
% FORMAT
% DESC returns the covariances of the smoothing distributions from the
% mutual information bound.
% ARG net : the network for which covariance is required.
% RETURN Cuu : covariance for first layer of weights.
% RETURN Cvv : covariance for second layer of weights.
% RETURN Cuv : cross covariance between layers.
% RETURN C : the full covariance of the smoothing distributions.
%
% SEEALSO : enscovar, smooth
%
% COPYRIGHT Mehdi Azzouzi and Neil D Lawrence, 1998, 1999

% ENSMLP
  
% Check arguments for consistency
type = {'smooth'};

if sum(strcmp(net.type, type)) == 0
  error('Undefined covariance structure. Exiting.');
else
  tnin = net.nin + 1;
  tnhidden = net.nhidden + 1;
  if nargout > 1
    nw1 = net.nhidden*(tnin);
    nw2 = (net.nhidden + 1)*net.nout;
    
    switch net.covstrct
      
     case 'diag' 			%Linear outputs
      
      C1 = diag(net.d1(:).^2);
      C2 = diag(net.d2(:).^2);
      C3 = zeros(nw1, nw2);
      
     case 'symmetric'
      nwts = nw1+nw2;
      void = ones(nwts);
      void = tril(void);
      C = zeros(nwts, nwts);
      C(find(void==1)) = net.U;
      C = C'*C;
      C1 = C(1:nw1, 1:nw1);
      C2 = C(nw1+(1:nw2), nw1+(1:nw2));
      C3 = C(1:nw1, nw1+(1:nw2));
      
     otherwise
      error('Covariance structure not yet implemented.\n');
      
    end
  end
end

% Collect the full covariance matrix
Covar = [C1 C3; C3' C2];






