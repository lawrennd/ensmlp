function [px, pm_x] = mixensdatall(net, x, t)


errstring = consist(net, 'mixens', x, t);
if ~isempty(errstring);
  error(errstring);
end

% Get the mixing coeff depending on the parametrisation
if strcmp(net.soft, 'y') == 1
  mixing_coeff = get_pi(net.z);
else
  mixing_coeff = net.pi;
end
ndata = size(x, 1);
% Calculate the portion of the data contribution due to each component
px = 0;
sigma = ones(ndata, 1);
pm_x = zeros(ndata, net.M);
px_m = zeros(ndata, net.M);
for m = 1:net.M
  [expy, expy2] = ensoutputexpec(net.ens(m), x);
  covar = ones(size(sigma))/net.ens(1).beta+(expy2-expy.^2);  
%  for i = 1:ndata
 %   px_m(i, m) = exp(-(ensdata_error(net.ens(m), x(i, :), t(i, :), ...
	%			     'beta')));
%  end
  px_m(:, m) = exp(-.5*((((expy-t).^2)./covar +net.nout*log(2*pi*covar))));
  px = px + mixing_coeff(m)*px_m(:, m);
end
for m = 1:net.M
  pm_x(:, m) = px_m(:, m)*mixing_coeff(m)./px;
end
 
