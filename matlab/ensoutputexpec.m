function [sexp, qexp] = ensoutputexpec(net, x, flag, datain, datatargets)

% ENSOUTPUTEXPEC gives the expectation of the output function and its square.
% FORMAT
% DESC gives the expectation of the output function and its square.
% ARG net : the network from which the expectations are taken.
% ARG x : the input locations for which the outputs are desired.
% RETURN y : the expected output associated with the given input.
% RETURN y2exp : the expectation of the square of the ouptut.
%
% COPYRIGHT : Neil D. Lawrence, 1998, 1999
%
% SEEALSO : ensoutputexpec

% ENSMLP
  
REAPPROX = 1;
LCONSTC = 1.160618371190047; % Lower approximation
UCONSTC = 1.273239525532378; % Upper approximation
ACONSTC = 1.238228217053698; % Approximation

CONSTC = ACONSTC; 

[Cuu, Cvv, Cuv] = enscovar(net);
fact = 1/sqrt(2);
V = [net.w2; net.b2];
U = [net.w1; net.b1];
tnhidden = net.nhidden + 1; % include the bias
tnin = net.nin + 1;
ndata = size(x, 1);
xTu = x*net.w1 + nrepmat(net.b1, 1, ndata);

quadexpec = 1;
if nargout < 2
  quadexpec = 0;
end

if quadexpec
  Theta = zeros(tnhidden, tnhidden, net.nout);
  qexp = zeros(ndata, net.nout); 
  g2 = zeros(ndata, net.nhidden);
end

Phi = zeros(ndata, net.nhidden);
g = zeros(ndata, net.nhidden);
aTa = zeros(ndata, 1);
store = zeros(ndata, tnin);
a0 = zeros(ndata, 1);
bTb = zeros(ndata, 1);
b0 = zeros(ndata, 1);
sexp = zeros(ndata, net.nout); 
nw1 = net.nin*net.nhidden;
nw2 = net.nhidden*net.nout;
% Could save time computationally by checking for diagonal here.
switch net.covstrct
case 'none'
  for row =1:net.nhidden
    a0(:, :) = xTu(:, row);
    g(:, row) = erf(fact*a0);
  end
  for p = 1:net.nout  
    sexp(:, p) = g * net.w2(:, p) + nrepmat(net.b2(1, p), 1, ndata);
    if quadexpec
      qexp(:, p) = sexp(:, p).^2;
    end
  end
  if nargin ==3
    if strcmp(flag, 'hess')
      [hess, dh] = enshess(net, datain, datatargets); 	% Full Hessian
      [evec, eval] = eig(dh);
      eval = eval.*(eval > 0);          % Remove -ve eigen values
      hess = enshess(net, datain, datatargets, dh); 	% Full Hessian
      invhess = inv(hess);
      expy2 = zeros(nplot, net.nout);
      [void, g] = ensoutputexpec(net, x);
      grad = ensexpgrad(net, x);
      for p = 1:net.nout
	for n = 1 : ndata
	  expy2(n, p) = grad(p, :, n)*invhess*grad(p, :, n)'+ expy(n)^2;
	end
      end
    end
  end
    
case {'diag', 'noded'}
  mark2 = 0;
  for row =1:net.nhidden
    mark1 = mark2;
    mark2 = mark1 + net.nin;
    rowrange = (mark1+1):mark2;
    biasrow = nw1 + row;
    aTa(:, :) = sum(x.*(x*Cuu(rowrange, rowrange)), 2) + ...
	2*x*Cuu(rowrange, biasrow) + ...
	nrepmat(Cuu(biasrow, biasrow), 1, ndata);
    a0 = xTu(:, row);
    % the expected value of the hidden node for each data point
    g(:, row) = apperf(fact*a0./sqrt(1+aTa));
    if quadexpec
      switch net.covstrct
      case 'none'
	g2(:, row) = g(:, row).*g(:, row);
	
      otherwise
	varequiv = (1+CONSTC*aTa);
	sdequiv = sqrt(varequiv);
	aTasquared = aTa.^2;
	% the expected value of the hidden node squared for each data point
	if REAPPROX
	  g2(:, row) = 1./sdequiv.*(apperf(fact*a0./sdequiv).^2+sdequiv-1);
	else
	  g2(:, row) = 1-1./sdequiv...
	      .*exp(-CONSTC*a0.^2.*(1./(2.*varequiv)));
	end
      end
    end
  end

  for p = 1:net.nout  
    sexp(:, p) = g * net.w2(:, p) + nrepmat(net.b2(1, p), 1, ndata);
  end

  if quadexpec
    % replace expected value of hidden node squared with variance
    gvar = g2 - g.*g;
    % Due to the approximation it is possible for gvar to be small and negative
    if max(max(gvar<0))
      gvar = gvar.*(gvar>=0);
      %fprintf(['Variance approximation is negative and being corrected in' ...
	%       ' ensoutputexpec. \n'])
    end
    mark2 = 0;
    for p = 1:net.nout
      mark1 = mark2;
      mark2 = mark1 + net.nhidden;
      rowrange = (mark1+1):mark2;
      biasrow = nw2 + p;
      tempcov = [Cvv(rowrange, rowrange) Cvv(rowrange, biasrow);
	  Cvv(biasrow, rowrange) Cvv(biasrow, biasrow)];
      Theta(:, :, p) = [V(:, p)*V(:, p)' + tempcov];
    end
    for p = 1:net.nout
      % First add the parts due to the mean^2 of the hidden nodes
      qexp(:, p) = sum(g.*(g*Theta(1:net.nhidden, 1:net.nhidden, p)), 2) ...
          + 2*g*Theta(1:net.nhidden, tnhidden, p) ...
          + nrepmat(Theta(tnhidden, tnhidden, p), 1, ndata);  
      % Next account for the variance of the hidden nodes
      qexp(:, p) = qexp(:, p) ...
          + gvar*diag(Theta(1:net.nhidden, 1:net.nhidden, p));  
    end
  end
otherwise
  error('Covariance function not yet implemented')
end


    












