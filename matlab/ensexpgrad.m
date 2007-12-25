function [sg, qg] = ensexpgrad(net, x)

% ENSEXPGRAD expectation of the gradient.
% FORMAT
% DESC expectation of the gradient.
% ARG net : the network.
% ARG x : the input location.
% RETURN sg : xx.
% RETURN qg : xx.
%
% COPYRIGHT Neil D. Lawrence, 1999
%
% SEEALSO : ens
  
% ENSMLP 
REAPPROX = 1;
LCONSTC = 1.160618371190047; % Lower approximation
UCONSTC = 1.273239525532378; % Upper approximation
ACONSTC = 1.238228217053698; % Approximation
CONSTC = ACONSTC;

fact = sqrt(2)/2;
w = enspak(net);
[Cuu, Cvv, Cuv] = enscovar(net);
V = [net.w2; net.b2];

tnhidden = net.nhidden + 1; % include the bias
tnin = net.nin + 1;

nnonbiasw1 = net.nin*net.nhidden;
nnonbiasw2 = net.nhidden*net.nout;

quadexp = 1;
if nargout < 2
  quadexp = 0;
end

qexpgrad = net;
sexpgrad = net;
ndata = size(x, 1);
for n = 1:ndata
  for p = 1:net.nout
    if quadexp
      Theta = zeros(tnhidden, tnhidden);
      Thetag = zeros(tnhidden);
      v = zeros(1, net.nhidden);
      g2 = zeros(net.nhidden, 1);
      g2dashbase = zeros(net.nhidden, 1);
      g2dash_w1 = zeros(net.nin, net.nhidden);
      g2dash_b1 = zeros(1, net.nhidden);
      
      qexpgrad.w1 = zeros(net.nin, net.nhidden);
      qexpgrad.b1 = zeros(1, net.nhidden);
      qexpgrad.w2 = zeros(net.nhidden, net.nout);
      qexpgrad.b2 = zeros(1, net.nout);
      
      switch net.covstrct
      case 'none'
	% Do nothing
	
      otherwise
    
	qexpgrad.d1 = zeros(net.nin, net.nhidden);
	qexpgrad.db1 = zeros(1, net.nhidden);
	qexpgrad.d2 = zeros(net.nhidden, net.nout);
	qexpgrad.db2 = zeros(1, net.nout);
	
      end
    end
    
    sexpgrad.w1 = zeros(net.nin, net.nhidden);
    sexpgrad.b1 = zeros(1, net.nhidden);
    sexpgrad.w2 = zeros(net.nhidden, net.nout);
    sexpgrad.b2 = zeros(1, net.nout);
    
    switch net.covstrct
    case 'none'
      % do nothing
      
    otherwise 
      sexpgrad.d1 = zeros(net.nin, net.nhidden);
      sexpgrad.db1 = zeros(1, net.nhidden);
      sexpgrad.d2 = zeros(net.nhidden, net.nout);
      sexpgrad.db2 = zeros(1, net.nout);
    end

    xTu = x(n, :)*net.w1 + net.b1;
    
    Phi = zeros(net.nhidden, 1);
    sqrtPhi = zeros(net.nhidden, 1);
    g = zeros(net.nhidden, 1);
    gdashbase = zeros(net.nhidden, 1);
    
    if quadexp
      mark1 = (p-1)*net.nhidden;
      mark2 = mark1 + net.nhidden;
      rowrange = (mark1+1):mark2;
      biasrow = nnonbiasw2 + p;
      tempcov = [Cvv(rowrange, rowrange) Cvv(rowrange, biasrow);
	  Cvv(biasrow, rowrange) Cvv(biasrow, biasrow)];
      Theta(:, :) = [V(:, p)*V(:, p)' + tempcov];
    end  
    
    aTa = zeros(1, 1);
    store = zeros(1, tnin);
    a0 = zeros(1, 1);
    
    
    % Could save time computationally by checking for diagonal here.
    mark2 = 0;
    for row =1:net.nhidden
      mark1 = mark2;
      mark2 = mark1 + net.nin;
      rowrange = (mark1+1):mark2;
      biasrow = nnonbiasw1 + row;
      aTa(row) = sum(x(n, :).*(x(n, :)*Cuu(rowrange, rowrange)), 2) + ...
	  2*x(n, :)*Cuu(rowrange, biasrow) + ...
	  Cuu(biasrow, biasrow);
      a0 = xTu(1, row);
      Phi(row) = 1+aTa(row);
      sqrtPhi(row) = sqrt(Phi(row));
      % the expected value of the hidden node squared for each data point
      % the expected value of the hidden node for each data point
      g(row) = apperf(fact*a0/sqrtPhi(row));
      gdashbase(row) = sqrt(2/pi)*exp(-(a0^2)/(2*Phi(row)));
      if quadexp    
	switch net.covstrct
	case 'none'
	  g2(row) = g(row)*g(row);
	  
	otherwise
	  varequiv(row) = 1 + CONSTC*aTa(row);
	  sdequiv(row) = sqrt(varequiv(row));
	  aTasquared(row) = aTa(row)^2;
	  % The expected value of each hidden node squared
	  if REAPPROX
	    g2(row) = 1./sdequiv(row) ...
		.*(apperf(fact*a0./sdequiv(row)).^2-1)+1;
	  else
	    g2(row) = 1-1/sdequiv(row)* ...
		      exp(-CONSTC*a0^2*(1/(2*varequiv(row))));
	    g2dashbase(row) = -g2(row) + 1;
	  end
	end
      end
    end
    
    % Calculate gradient of hidden units expectation with respect
    % to first layer weights and biases
    for h = 1:net.nhidden
      gdash_b1(1, h) = gdashbase(h)/sqrtPhi(h);
      gdash_w1(:, h) = x(n, :)'*gdash_b1(1, h);
    end

    % calculate gradient of output expectation with respect to
    % first layer weights and biases 
    for h = 1:net.nhidden
      sexpgrad.w1(:, h) = gdash_w1(:, h).*net.w2(h, p);
      sexpgrad.b1(:, h) = gdash_b1(:, h).*net.w2(h, p);
    end  
    
    % calculate gradient of output expectation with respect to
    % second layer weights and biases
    sexpgrad.w2(:, p) = g;
    sexpgrad.b2(1, p) = 1;
    
    
    % check if there are other parameters whose gradients are
    % required
    switch net.covstrct
    case 'none'
      % do nothing
      
    otherwise
      % calculate gradient of hidden layer expectation with respect
      % to first layer covariance parameters d
      gdash_dw1 = zeros(net.nin, net.nhidden);
      gdash_db1 = zeros(1, net.nhidden);
      x2d1 = zeros(net.nin, net.nhidden);
      gTvdash_d1 = zeros(net.nin + 1, net.nhidden);
      gdash_db1 = -xTu./(Phi.*sqrtPhi)'.*gdashbase';
      for h = 1:net.nhidden
	x2d1(:, h) =  (x(n, :).*x(n, :))'.*net.d1(:, h); 
	gdash_dw1(:, h) = x2d1(:, h).*gdash_db1(1, h);
      end  
      gdash_db1 = gdash_db1.*net.db1;
      
      % calculate gradient of output expectation with respect to
      % first layer covariance parameters d      
      gdash_d1 = [gdash_dw1; gdash_db1];
      for h=1:net.nhidden
	sexpgrad.d1(:, h) = gdash_dw1(:, h).*net.w2(h, p);
	sexpgrad.db1(1, h) = gdash_db1(1, h).*net.w2(h, p);
      end
      
      switch net.covstrct
      case 'diag'
	% do nothing
	
      otherwise
	error('Covariance function not yet implemented')
      end
    end
    
    % Now check if the gradients of the
    % expectation of the square are also required
    if quadexp
      
      gTv = (g'*net.w2(:, p)) + net.b2(1, p);
      v = [net.w2(:, p); net.b2(1, p)];
      Thetag = Theta(:, 1:net.nhidden)*g + Theta(:, tnhidden);  
      
      switch net.covstrct
      case 'none'
	qexpgrad.w2(:, p) = 2*gTv.*g;
	qexpgrad.b2(1, p) = 2*gTv;
	
      otherwise 				
	gvar = g2 - g.*g;
	% Due to the approximation it is possible for gvar to be small and -ve
	negativenodes = (gvar<0);
	if max(max(negativenodes))
	  gvar = gvar.*~negativenodes;
	  %fprintf(['Variance approximation is negative and being corrected' ...
		%  'in ensexpgrad \n'])
	end
	
	qexpgrad.w2(:, p) = 2*gTv.*g+ 2*gvar.*net.w2(:, p);
	qexpgrad.b2(1, p) = 2*gTv;
	
	qexpgrad.d2(:, p) = 2*net.d2(:, p).*(g.*g + gvar);
	qexpgrad.db2(1, p) = 2*net.db2(1, p); 
	
	switch net.covstrct
	case 'diag'
	  % do nothing
	  
	otherwise
	  error('Covariance function not yet implemented')
	end
      end
      
      % Strange, I had to put an i in qexpgrad.w1(i, :)
      for i = 1:(net.nin)
	qexpgrad.w1(i, :) = 2*Thetag(1:net.nhidden)'.*gdash_w1(i, :);
      end
      qexpgrad.b1(:, :) = 2*Thetag(1:net.nhidden)'.*gdash_b1(1, :);
       
      switch net.covstrct
      case 'none'
	
      otherwise
	% calculate gradient of the expectation of the hidden layer
	% squared with respect to first layer weights and biases
	if REAPPROX
	  for h = 1:net.nhidden
	    if negativenodes(h)
	      g2dash_w1(:, h) = 2*g(h).*gdash_w1(:, h);
	      g2dash_b1(1, h) = 2*g(h).*gdash_b1(1, h);
	    else	      
	      argument = xTu(h)/sdequiv(h);
	      g2dash_b1(1, h) = 1/varequiv(h)...
		  *(2*apperf(fact*argument)*sqrt(2/pi)...
		    * exp(-.5*argument*argument));
	      g2dash_w1(:, h) = (x(n, :)*g2dash_b1(1, h))';
	    end
	  end
	else
	  for h = 1:net.nhidden
	    if negativenodes(h)
	      g2dash_w1(:, h) = 2*g(h).*gdash_w1(:, h);
	      g2dash_b1(1, h) = 2*g(h).*gdash_b1(1, h);
	    else
	      g2dash_b1(1, h) = CONSTC*xTu(h)*(1/varequiv(h))*g2dashbase(h);
	      g2dash_w1(:, h) = x(n, :)*g2dash_b1(1, h);
	    end
	  end
	end
	for i = 1:net.nin   
	  gvardash_w1(i, :) = g2dash_w1(i, :) - 2*g'.*gdash_w1(i, :);
	end
	gvardash_b1(1, :) = g2dash_b1(1, :) - 2*g'.*gdash_b1(1, :);
	for i = 1:(net.nin)
	  gThetagvardash_w1(i, :) = ...
	      diag(Theta(1:net.nhidden, 1:net.nhidden))'.*gvardash_w1(i, :);
	end
	gThetagvardash_b1(1, :) = ...
	    diag(Theta(1:net.nhidden, 1:net.nhidden))'.*gvardash_b1(1, :);
	
	qexpgrad.w1 = qexpgrad.w1 + gThetagvardash_w1;
	qexpgrad.b1 = qexpgrad.b1 + gThetagvardash_b1;

	% calculate gradient of the expectation of the hidden layer
        % squared with respect to first layer covariance parameters d
	g2dash_dw1 = zeros(net.nin, net.nhidden);
	g2dash_db1 = zeros(1, net.nhidden);
	if REAPPROX
	  for h = 1:net.nhidden
	    if negativenodes(h)
		g2dash_dw1(:, h) = 2*g(h).*gdash_dw1(:, h);
		g2dash_db1(1, h) = 2*g(h).*gdash_db1(1, h);
	    else	   
	      argument = xTu(h)/sdequiv(h);
	      tempfactor = (g2(h)-1) ...
		  +xTu(h)/(varequiv(h))...
		  *(2*apperf(fact*argument)*sqrt(2/pi)...
		  * exp(-.5*argument*argument));
	      tempfactor = -CONSTC*tempfactor/varequiv(h);
	      g2dash_dw1(:, h) = x2d1(:, h) * tempfactor;
	      g2dash_db1(1, h) = net.db1(1, h) * tempfactor;
	    end
	  end
	else
	  for h = 1:net.nhidden
	    if negativenodes(h)
		g2dash_dw1(:, h) = 2*g(h).*gdash_dw1(:, h);
		g2dash_db1(1, h) = 2*g(h).*gdash_db1(1, h);
	    else
	      tempfactor = CONSTC*(1/varequiv(h)-CONSTC*(xTu(h)*xTu(h))...
				   /(varequiv(h)*varequiv(h)))*g2dashbase(h);
	      g2dash_dw1(:, h) = x2d1(:, h) * tempfactor;
	      g2dash_db1(1, h) = net.db1(1, h) * tempfactor;
	    end
	  end
	end
	
	
	% Add the gradient of the variance of the output squared to
        % the gradient of the squared expectation of the output
	for i = 1:net.nin
	  gvardash_d = g2dash_dw1(i, :) - 2*g'.*gdash_dw1(i, :);
	  qexpgrad.d1(i, :) = 2*Thetag(1:net.nhidden)'.*gdash_dw1(i, :) ...
	      + diag(Theta(1:net.nhidden, 1:net.nhidden))'...
	      .* gvardash_d;
	end
	gvardash_d = g2dash_db1(1, :) - 2*g'.*gdash_db1(1, :);
	qexpgrad.db1(1, :) = 2*Thetag(1:net.nhidden)'.*gdash_db1(1, :) ...
	    + diag(Theta(1:net.nhidden, 1:net.nhidden))' ...
	    .* gvardash_d;
	
	switch net.covstrct
	case 'diag'
	  % do nothing
	  
	otherwise
	  error('Covariance function not yet implemented')
	end 
      end  
      qg(p, :, n) = enspak(qexpgrad);
      
    end
    sg(p, :, n) = enspak(sexpgrad);
    
  end
end
