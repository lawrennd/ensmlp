function g = ensdata_grad(net, x, t);

% ENSDATA_GRAD Gradient of the data portion.
% FORMAT
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1999

% ENSMLP
  
tnin = 1 + net.nin;
tnhidden = 1 + net.nhidden;
ndata = size(x, 1);
gnet = net;
gnet.w1 = zeros(net.nin, net.nhidden);
gnet.b1 = zeros(1, net.nhidden);
gnet.d1 = zeros(net.nin, net.nhidden);
gnet.db1 = zeros(1, net.nhidden);
gnet.w2 = zeros(net.nhidden, net.nout);
gnet.b2 = zeros(1, net.nout);
gnet.d2 = zeros(net.nhidden, net.nout);
gnet.db2 = zeros(1, net.nout);

if isfield(net, 'beta')
  if size(net.beta) == [1 1]
    vbeta = ones(net.nout, 1)*net.beta;
  else
    vbeta = net.beta;
  end
else
  vbeta = ones(net.nout, 1);
end
[sg, qg] = ensexpgrad(net, x);

for n = 1:ndata
  gexpty = zeros(1, net.npars);
  gexpy2 = zeros(1, net.npars);

  for p = 1:net.nout
    gexpty = gexpty + sg(p, :, n)*vbeta(p)*t(n, p);
    gexpy2 = gexpy2 + qg(p, :, n)*vbeta(p);
  end    
  
  exptygrad = ensunpak(net, gexpty);  
  expy2grad = ensunpak(net, gexpy2);  

  for h = 1:net.nhidden	
    for i = 1:net.nin
      gnet.w1(i, h) = gnet.w1(i, h) + expy2grad.w1(i, h) ...
	  - 2*exptygrad.w1(i, h, p);
    end
    gnet.b1(1, h) = gnet.b1(1, h) + expy2grad.b1(1, h)...
	  - 2*exptygrad.b1(1, h);
    for p = 1:net.nout 
      gnet.w2(h, p) = gnet.w2(h, p) + expy2grad.w2(h, p) ...
	  - 2*exptygrad.w2(h, p);
    end
  end
  
  for p = 1:net.nout
    gnet.b2(1, p) = gnet.b2(1, p) + expy2grad.b2(1, p) ...
        - 2*exptygrad.b2(1, p);
  end

  switch net.covstrct
  case 'none'
    % do nothing

  otherwise
    for h = 1:net.nhidden	
      for i = 1:net.nin
        gnet.d1(i, h) = gnet.d1(i, h) + expy2grad.d1(i, h) ...
	    - 2*exptygrad.d1(i, h, p);
      end
      gnet.db1(1, h) = gnet.db1(1, h) + expy2grad.db1(1, h)...
	  - 2*exptygrad.db1(1, h);
      for p = 1:net.nout 
        gnet.d2(h, p) = gnet.d2(h, p) + expy2grad.d2(h, p) ...
            - 2*exptygrad.d2(h, p);
      end
    end
  
    for p = 1:net.nout
      gnet.db2(1, p) = gnet.db2(1, p) + expy2grad.db2(1, p) ...
        - 2*exptygrad.db2(1, p);
  end
   
    
    switch net.covstrct
    
    case 'diag'
      % do nothing
      
    otherwise      
      error('Covariance function not yet implemented')
    end
  end
end
g = .5*enspak(gnet);








