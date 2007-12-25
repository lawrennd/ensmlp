function [grad] = mmigrad(net, calcQint)
% MMIGRAD Computes the gradient of the MI with respect to the parameters of the net
% FORMAT
% DESC returns gradients of the mutual information with respect to
% several parameters of the network.
% ARG net : the network.
% RETURN gmixing : the gradient with respect to the mixing coef. of the mixture Q.
% RETURN glambda : the gradient with respect to the mixing coef. of the mixture R.
% RETURN gq : the gradient with respect to the parameters of Q.
% RETURN gr : the gradient with respect to the parameters of R
%          
%          
% SEEALSO : MMI, GET_PI, GET_LAMBDA, TRACEQR, DISTR, SUMDET
%          
% COPYRIGHT : Mehdi Azzouzi and Neil D. Lawrence, 1998, 1999
  
% ENSMLP
          
% Check type of net
if strcmp(net.type, 'mixens') == 0 & strcmp(net.type, 'mmi_R') == 0
  error('Undefined type of net.\n');
end
if nargin == 1;
  calcQint = 1;
end

% Number of mixtures
M = net.M;
% Number of weights for Q
nwts = net.ens(1).nwts;
nw1 = net.ens(1).nhidden*net.ens(1).nin; 	%number of weights in layer 1
nb1 = net.ens(1).nhidden; 			%number of biases in layer 1
nw2 = net.ens(1).nhidden*net.ens(1).nout; 	%number of weights in layer 2
nb2 = net.ens(1).nout;  			%number of biases in layer 2
np1 = nb1 + nw1;        			%number of parameters in layer 1
np2 = nb2 + nw2;        	 		%number of parameters in layer 2

% Number of weights for R
tnin = net.smooth(1).nin + 1;
tnhidden = net.smooth(1).nhidden + 1;
nw1_R = net.smooth(1).nhidden*(tnin);
nw2_R = (net.smooth(1).nhidden + 1)*net.smooth(1).nout;
%if strcmp(net.smooth(1).covstrct, 'unnoded') == 1
%  S = net.smooth(1).t; 			        %mumber of unnoded terms
%end
% Get the mixing coefficients for Q and R
if strcmp(net.soft, 'y') == 1
  mixing = get_pi(net.z);
else
  mixing = net.pi;
end

lambda = get_lambda(net.y);

% Gradients wrt z and y
if strcmp(net.soft, 'y') == 1
  grad_mix = zeros(1, M);
  grad_mix_z = diag(mixing) - mixing'*mixing;
end

grad_lambda = zeros(1, M);
grad_lambda_y = lambda;

% Gradients wrt vecQ and vecR
grad_vecR = zeros(nwts, M);

% Gradient wrt CovarQ
switch net.ens(1).covstrct
 case 'none'
  % do nothing
  
 case 'diag'
  if calcQint
    grad_covQ = zeros(nwts, M);
  end
  for m = 1:M
    CovarQ{m} = [net.ens(m).d1(:); net.ens(m).db1(:); 
		 net.ens(m).d2(:); net.ens(m).db2(:)];
    CovarQ{m} = sparse(diag(CovarQ{m}.*CovarQ{m}));
  end
  
 otherwise
  error('Ensemble Covariance function not yet implemented.\n')
  for m = 1:M
    [void, void, void, CovarQ{m}] = enscovar(net.ens(m));
  end
end


% Set up parameters and distributions for R
switch net.smooth(1).covstrct
  case 'diag'
    grad_covR = zeros(nwts, M);
    for m = 1:M
      CovarR{m} = [net.smooth(m).d1(:); net.smooth(m).d2(:)];
      CovarR{m} = CovarR{m}.*CovarR{m};
      InverseR{m} = 1./CovarR{m};
      CovarR{m} = sparse(diag(CovarR{m}));
      InverseR{m} = sparse(diag(InverseR{m}));
    end
    
 case 'symmetric'
  grad_covR = zeros(nwts*(nwts+1)/2, M);
  for m = 1:M
    [void, void, void, CovarR{m}] = smoothcovar(net.smooth(m));
    [void, void, void, InverseR{m}] = smoothcovinv(net.smooth(m));
  end
 otherwise
  error('Smooth Covariance function not yet implemented.\n')
end

% Get the parameters for Q and R

% Now enter the main loop
% Loop over the mixing coeff. of Q
for k = 1:M
  

  tr = traceQR(net.ens(k), net.smooth(k), CovarQ{k}, InverseR{k});
  [dist, vecQ, vecR, diff_vecQ_vecR]= distR(net.ens(k), ...
					    net.smooth(k), ...
					    CovarQ{k}, ...
					    InverseR{k});

  if strcmp(net.soft, 'y') == 1
    grad_mix(k) = - .5 * (tr + dist);
  end
  
  grad_vecR(:, k) = mixing(k) * InverseR{k} * diff_vecQ_vecR;
  if calcQint
    grad_vecQ(:, k) = -grad_vecR(:, k);
  end

  % Get the original elements of CovarQ
  if calcQint
    C1Q = [net.ens(k).d1(:)' net.ens(k).db1];
    C2Q = [net.ens(k).d2(:)' net.ens(k).db2];
    CQ = [C1Q C2Q]';
    grad_covQ(:, k) = -.5*mixing(k)*diag(InverseR{k}).*(2*CQ);
  end
  % Get the original elements of CovarR
  switch net.smooth(k).covstrct
   case 'diag'
    CR = [net.smooth(k).d1(:)' net.smooth(k).d2(:)']';
    grad_covR(:, k) = .5 * mixing(k) * InverseR{k} * ...
	diag((CovarQ{k} + (diff_vecQ_vecR) * (diff_vecQ_vecR)') ...
	     * InverseR{k}) .* (2*CR);
    err_covR = zeros(nwts, 1);
     
   case 'symmetric'
    % Get the upper triangular matrix U
    U = zeros(nwts, nwts);
    void = ones(nwts);
    void = tril(void);
    U(find(void==1)) = net.smooth(k).U;
    U = U';
    A = .5 * mixing(k) * InverseR{k} * ...
	(CovarQ{k} + diff_vecQ_vecR * diff_vecQ_vecR') * InverseR{k} * (2*U);
    % Keep just the upper triangular part
    elements = tril(ones(size(A)));
    A = triu(A)';
    grad_covR(:,k) = A(elements~=0);
    err_covR = zeros(nwts*(nwts+1)/2, 1);
  end
  
  errR = zeros(nwts,1);
  % Loop over the mixing coeff. of R
  for m = 1:M
    
    % collect and compute some parameters needed in the derivations
    [suminv, suminv2, detprod] = ...
	sumdet(net.ens(k), net.smooth(m), CovarQ{k}, CovarR{m});
    invdetprod = 1 / sqrt(detprod);
    
    [dist, vecQ, vecR, diff_vecQ_vecR] = ...
	distsum(net.ens(k), net.smooth(m), suminv);
    term = invdetprod*exp(-.5*dist);

    if strcmp(net.soft, 'y') == 1
      grad_mix(k) = grad_mix(k) - lambda(m)*term;
    end
    grad_lambda(m) = grad_lambda(m) - mixing(k)*term;
    if calcQint
      grad_vecQ(:, k) = grad_vecQ(:, k) + mixing(k) * lambda(m) * ...
	  term * suminv * diff_vecQ_vecR;
      %termmat1 = suminv; %2 * InverseR{m};
      termmat2 = suminv * diff_vecQ_vecR * diff_vecQ_vecR' * suminv;
      termmatQ = suminv - termmat2;
    
      grad_covQ(:, k) = grad_covQ(:, k) + ...
	  .5 * mixing(k) * lambda(m) * term * diag(termmatQ) .* (2 * CQ);
    end
    
    % collect and compute some parameters needed in the derivations
    [suminv, suminvI, detprod] = ...
	sumdet(net.ens(m), net.smooth(k), CovarQ{m}, CovarR{k});
    invdetprod = 1 ./ sqrt(detprod);
    
    [dist, vecQ, vecR, diff_vecQ_vecR] = distsum(net.ens(m), ...
						 net.smooth(k), ...
						 suminv);
    term = invdetprod*exp(-.5*dist);
    %    termmat1 = suminvI * InverseR{k}*CovarQ{m}*InverseR{k};
    termmat1 = suminv*CovarQ{m}*InverseR{k};
    termmat2 = suminv * diff_vecQ_vecR * diff_vecQ_vecR' * suminv;
    termmatR = termmat1 + termmat2;

    errR = errR - mixing(m) * term * suminv * diff_vecQ_vecR;  
 
    switch net.smooth(k).covstrct
    
     case 'diag'
      CR = [net.smooth(k).d1(:)' net.smooth(k).d2(:)']';
      err_covR = err_covR - .5 * mixing(m) * term * diag(termmatR) .* (2 * CR);

     case 'symmetric'
      A = termmatR*(2*U);
      elements = tril(ones(size(A))); 
      A = triu(A)';
      A = A(elements~=0);
      err_covR = err_covR - .5 * mixing(m) * term * A;
      
    end % switch statement
  end % loop over m

  grad_vecR(:, k) = grad_vecR(:, k) + lambda(k)*errR;
  grad_covR(:, k) = grad_covR(:, k) + lambda(k)*err_covR;

end


grad_lambda = grad_lambda + mixing./lambda;

% Now compute the gradient with respect to z
grad_z = zeros(1, M);
if strcmp(net.soft, 'y') == 1
  % Final term of the MMI involving \alpha and \lambda
  grad_mix = grad_mix - mylog(mixing ./ lambda) - 1;
  for i = 1:M
    for j = 1:M
      grad_z(i) = grad_z(i) + grad_mix(j)*grad_mix_z(j, i);
    end
  end
end

% And the gradient with respect to y
grad_y = zeros(1, M);
grad_y = grad_lambda_y .* grad_lambda;

if strcmp(net.soft, 'y') == 1
  % Final term of the MMI involving \alpha and \lambda
  grad = [grad_z grad_y];
else
  grad = [grad_y];
end

% Reshape grad_z, grad_y, grad_vecRQ, grad_vecR, grad_covQ and grad_covR to a vector
if calcQint
  for m = 1:M
    grad = [grad grad_vecQ(:,m)' grad_covQ(:,m)' grad_vecR(:,m)' ...
	    grad_covR(:,m)'];
  end
else
  for m = 1:M
    grad = [grad grad_vecR(:,m)' ...
	    grad_covR(:,m)'];
  end
end  



