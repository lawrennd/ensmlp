function error = ensprior_error(net)

% ENSPRIOR_ERROR The prior term's portion of the error.
% FORMAT 
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP

if isfield(net, 'alphaprior')
  
  w = enspak(net);
  [expA, explogA] = priorinvcov(net);  
  
  switch net.covstrct
    
   case 'none'
    TrCA = 0;
    
   case 'diag'     
    diagC = [net.d1(:); net.db1(:); net.d2(:); net.db2(:)];
    diagC=diagC.*diagC;
    TrCA = diagC'*expA;
    
   otherwise
    error('Covariance type not yet implemented')
    
  end 
  
  error = 0.5*(w(1:net.nwts).^2*expA + TrCA ...
	    + net.nwts*log(2*pi)-sum(explogA));
else
  error = 0;
end


