function g = ensprior_grad(net)

% ENSPRIOR_GRAD Prior term's gradient.
% FORMAT 
% DESC returns the gradient of the prior part.
% ARG net : the network for which gradient is required.
% ARG x : input locations.
% RETURN g : the gradient of the prior portion.
%
% SEEALSO : ensgrad, ensentropy_grad
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP

if isfield(net, 'alpha')
  w = enspak(net);
  A = priorinvcov(net);    
  
  switch net.covstrct    
  case 'none'
    g = [A'.*w(1:net.nwts)];
    
  case 'diag'       
    g = [A'.*w(1:net.nwts), A'.*w(net.nwts+(1:net.nwts))];
    
  case 'noded'
    error('Code for noded not yet implemented')
      %g = net.alpha* ...
      %[w(1:ndet.nwts) 2*w((net.nwts+1):net.npars)];
  
  case 'layered'      
    error('Code for layered not yet implemented')
    
  case 'full'
      error('Code for full not yet implemented')
	
  end 
else
  g = 0;
end
