function h=ensentropy_error(net)

% ENSENTROPY_ERROR Entropy term's contribution to the error.
% FORMAT 
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP
  
switch net.covstrct

 case 'none'
  h = 0;
  
 case 'diag'     
  sqrtdiagC = abs([net.d1(:); net.db1(:); net.d2(:); net.db2(:)]);
  h = sum(log(sqrtdiagC), 1) + ...
      0.5*net.nwts*(1 + log(2*pi));
  
 otherwise
  [Cuu, Cvv, Cuv] = enscovar(net);
  C = [Cuu Cuv; Cuv' Cvv];
  error('Covariance function not yet implemented')
  
end 
