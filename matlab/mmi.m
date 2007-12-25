function err = mmi(net)

% MMI The bound on the mutual information term from a mixture of Gaussians.
% FORMAT
% DESC computes the bound on the mutual information term from a mixture of Gaussians.
% ARG net : the network containing the relevant mixture of Gaussians.
% RETURN mi : the mutual information of the mixture.
%
% SEEALSO : MIXENS, MIXENSPAK, MIXENSUNPAK, GET_PI, GET_LAMBDA, TRACEQR, DISTR, SUMDET
% 
% COPYRIGHT :  Mehdi Azzouzi and Neil D Lawrence, 1998, 1999

% ENSMLP
  
%/~%p = mixenspak(net);
%p = x;

%net = mixensunpak(net, p);
%~/
  
% Check type of net
if strcmp(net.type, 'mixens') == 0 & strcmp(net.type, 'mmi_R') == 0
  error('Undefined type of net.');
end

err = 0;
% Get the number of mixtures
M = net.M;

% Get the mixing coefficients
% Check the type of softmax
if strcmp(net.soft, 'y') == 1
  mixing = get_pi(net.z);
else
  mixing = net.pi;
end
 
lambda = get_lambda(net.y);

for m = 1:M
  
  % Get the parameters for Q and R
  [CuuQ, CvvQ, CuvQ, CovarQ{m}] = enscovar(net.ens(m));
  [CuuR, CvvR, CuvR, CovarR{m}] = smoothcovar(net.smooth(m));
  [IuuR, IvvR, IuvR, InverseR{m}] = smoothcovinv(net.smooth(m));

end
% Loop over the mixing coeff. of R
for m = 1:M
  

  % Get trace(Q(m)*inv(R(m))
  tr = traceQR(net.ens(m), net.smooth(m), CovarQ{m}, InverseR{m});
  
  % Get the distance with respect to R: 
  % (\mu_m-\delta_m)'*inv(Rm)*(\mu_m-\delta_m)
  dist = distR(net.ens(m), net.smooth(m), CovarQ{m}, InverseR{m});
  
  err = err - .5 * mixing(m) * (tr + dist);
  
  % Loop over the mixing coeff. of Q
  term = 0; 
  for n = 1:M    
    % Get inv(Q_n + R_m) and det(I + Q_n*inv(R_m))
    [suminv, suminv2, detprod] = ...
	sumdet(net.ens(n), net.smooth(m), CovarQ{n}, CovarR{m});
    invdetprod = 1 ./ sqrt(detprod);
    if ~isreal(invdetprod)
      warning('INVDETPROD is not real')
    end

    % Get the distance with respect to (Q+R): 
    % (\mu_n-\delta_m)'*inv(Q_n+R_m)*(\mu_n-\delta_m)
    dist = distsum(net.ens(n), net.smooth(m), suminv);
    term = term - mixing(n)*invdetprod*exp(-.5*dist);
    %term = term - mixing(n)*invdetprod;
  end

  err = err + lambda(m)*term;
end

%  Final term containing the mixing coeff. of Q and R
%err = err - sum(mixing.*(log(mixing./lambda))) + 1;
err = err - sum(xlogy(mixing)) + sum(xlogy(mixing, lambda)) + 1;