function [net] = mixensmixmstep(net, x, t)

% MIXENSMIXMSTEP re-estimate the mixing coefficients of the mixture.
% FORMAT               
% DESC performs a re-estimation step of the mixing coefficients.
% ARG net : the network for which reestimation is to be performed.                
% ARG x : the input data locations.
% ARG t : the output dat alocations.
% RETURN net : the returned network with updated mixture coefficients.
%
% SEEALSO : mixens, mmi, grad
%                
% COPYRIGHT : Mehdi Azzouzi and Neil D Lawrence,1998, 1999
  
% ENSMLP
 
M = net.M;
num = zeros(1,M);
lambda = get_lambda(net.y);

for m = 1:M
  num(m) = num(m) + ensentropy_error(net.ens(m)) - ...
      ensdata_error(net.ens(m), x, t, 'beta');
  if isfield(net.ens(m), 'alphaprior')
    num(m) = num(m) - ensprior_error(net.ens(m));
  end
  
  % Get the parameters for Q and R
  [CuuQ, CvvQ, CuvQ, CovarQm] = enscovar(net.ens(m));
  [CuuR, CvvR, CuvR, CovarRm] = smoothcovar(net.smooth(m));
  [IuuR, IvvR, IuvR, InverseRm] = smoothcovinv(net.smooth(m));

  % Get trace(Q(m)*inv(R(m))
  tr = traceQR(net.ens(m), net.smooth(m), CovarQm, InverseRm);
  
  % Get the distance with respect to R: 
  % (\mu_m-\delta_m)'*inv(Rm)*(\mu_m-\delta_m)
  dist = distR(net.ens(m), net.smooth(m), CovarQm, InverseRm);
  
  num(m) = num(m) - .5 * (tr + dist);

  % Loop over the mixing coeff. of R
  term = 0; 
  for n = 1:M    
    % Get the parameter of Q_n
    [CuuR, CvvR, CuvR, CovarRn] = smoothcovar(net.smooth(n));
    %[CuuQ, CvvQ, CuvQ, CovarQn] = enscovar(net.ens(n));
    % Get inv(Q_n + R_m) and det(I + Q_n*inv(R_m))
    [suminv, suminv2, detprod] = ...
	sumdet(net.ens(m), net.smooth(n), CovarQm, CovarRn);
    invdetprod = 1 ./ sqrt(detprod);
    if ~isreal(invdetprod)
      warning('INVDETPROD is not real')
    end

    % Get the distance with respect to (Q+R): 
    % (\mu_n-\delta_m)'*inv(Q_n+R_m)*(\mu_n-\delta_m)
    dist = distsum(net.ens(m), net.smooth(n), suminv);
    term = term - lambda(n)*invdetprod*exp(-.5*dist);
  end
  
  num(m) = num(m) + term + log(lambda(m));
end

newnum = num-max(num);
%sumnewnum = sum(newnum);
net.pi = exp(newnum)/sum(exp(newnum));
%for m = 1:M
%  a = 0;
%  % Sum up all the terms but m
%  for n = 1:M
%    if n ~= m
%      a = a + num(n);
%    end
%  end

%  % Substract from num this sum
%  num2(m) = num(m) - a;
%end
%a
%net.pi = 1 ./ (1 + exp(num2));
