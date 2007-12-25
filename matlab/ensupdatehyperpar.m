function net = ensupdatehyperpar(net, x, t)

% ENSUPDATEHYPERPAR Re-estimate parameters of the hyper posteriors.
% FORMAT
% DESC re-estimates the hyperparameters
% ALPHA and BETA. PRIORS is a vector of the hyperpriors parameters, a,
% b, c, and d.
%
% ARG net : the network for which hyper parameters are estimated.
% ARG x : the input locations of the network.
% ARG t : the target locations.
% RETURN net : network with hyper parameters updated.
%
% SEEALSO : ENSHYPERPRIOR, ENSGRAD, ENSERR, DEMENS1
%
% COPYRIGHT :  Neil D Lawrence, Mehdi Azzouzi (1999)

% ENSMLP

a_alpha = net.alphaprior.a;
b_alpha = net.alphaprior.b;

exp2Net = net;
switch net.covstrct
 case 'none'
  hess = enshess(net, x, t); 		% Full Hessian
  tempNet = ensunpak(net, diag(hess));    
  
  exp2Net.w1 = tempNet.w1 + net.w1.^2;
  exp2Net.b1 = tempNet.b1 + net.b1.^2;
  exp2Net.w2 = tempNet.w2 + net.w2.^2;
  exp2Net.b2 = tempNet.b2 + net.b2.^2;
 case 'diag'    
  exp2Net.w1 = (net.d1.^2 + net.w1.^2);
  exp2Net.b1 = (net.db1.^2 + net.b1.^2);
  exp2Net.w2 = (net.d2.^2 + net.w2.^2);
  exp2Net.b2 = (net.db2.^2 + net.b2.^2);
  
 otherwise
  error('Covariance function not yet implemented')
end 


switch net.alphaprior.type
case 'single'
  % hyper prior is single value
  [Cuu, Cvv, Cuv] = enscovar(net);
  w = enspak(net);
  % Update the parameters of Q(\alpha)
  abar_alpha = a_alpha + net.nwts/2;  
  bbar_alpha = b_alpha + .5*sum(sum(exp2Net.w1)) + .5*sum(exp2Net.b1) ...
       + .5*sum(sum(exp2Net.w2)) + .5*sum(exp2Net.b2);
case 'group'
  % Weights and biases are grouped seperately
  mark1 = 1;
  mark2 = mark1 + 1;
  mark3 = mark2 + 1;
  mark4 = mark3 + 1;
  a_alphaI = a_alpha(1:mark1);
  a_alphaIb = a_alpha(mark1+1:mark2);
  a_alphaH = a_alpha(mark2+1:mark3);
  a_alphaHb = a_alpha((mark3+1):mark4);
  b_alphaI = b_alpha(1:mark1);
  b_alphaIb = b_alpha(mark1+1:mark2);
  b_alphaH = b_alpha(mark2+1:mark3);
  b_alphaHb = b_alpha((mark3+1):mark4);
  abar_alphaI = a_alphaI + net.nin*net.nhidden/2;
  abar_alphaIb = a_alphaIb + net.nhidden/2;
  abar_alphaH = a_alphaH + net.nhidden*net.nout/2;
  abar_alphaHb = a_alphaHb + net.nout/2;
  bbar_alphaI = b_alphaI;
  bbar_alphaIb = b_alphaIb;
  bbar_alphaH = b_alphaH;
  bbar_alphaHb = b_alphaHb;
  
  bbar_alphaI = b_alphaI + sum(sum(exp2Net.w1, 2), 1)/2; 
  bbar_alphaIb = b_alphaIb + sum(exp2Net.b1, 2)/2; 
  bbar_alphaH = b_alphaH + sum(sum(exp2Net.w2, 2), 1)/2; 
  bbar_alphaHb = b_alphaHb + sum(exp2Net.b2, 2)/2; 
  abar_alpha = [abar_alphaI; abar_alphaIb; ...
		abar_alphaH; abar_alphaHb];
  bbar_alpha = [bbar_alphaI; bbar_alphaIb; ...
		bbar_alphaH; bbar_alphaHb];

case 'ARD'
  mark1 = net.nin;
  mark2 = mark1 + 1;
  mark3 = mark2 + 1;
  mark4 = mark3 + 1;
  a_alphaI = a_alpha(1:mark1);
  a_alphaIb = a_alpha(mark1+1:mark2);
  a_alphaH = a_alpha(mark2+1:mark3);
  a_alphaHb = a_alpha((mark3+1):mark4);
  b_alphaI = b_alpha(1:mark1);
  b_alphaIb = b_alpha(mark1+1:mark2);
  b_alphaH = b_alpha(mark2+1:mark3);
  b_alphaHb = b_alpha((mark3+1):mark4);

  abar_alphaI = a_alphaI + net.nhidden/2;
  abar_alphaIb = a_alphaIb + net.nhidden/2;
  abar_alphaH = a_alphaH + net.nhidden*net.nout/2;
  abar_alphaHb = a_alphaHb + net.nout/2;
 
  bbar_alphaI = b_alphaI + sum(exp2Net.w1, 2)/2; 
  bbar_alphaIb = b_alphaIb + sum(exp2Net.b1, 2)/2; 
  bbar_alphaH = b_alphaH + sum(sum(exp2Net.w2, 2), 1)/2; 
  bbar_alphaHb = b_alphaHb + sum(exp2Net.b2, 2)/2; 
 
  abar_alpha = [abar_alphaI; abar_alphaIb; ...
		abar_alphaH; abar_alphaHb];
  bbar_alpha = [bbar_alphaI; bbar_alphaIb; ...
		bbar_alphaH; bbar_alphaHb];

case 'NRD'
  % A hyperprior is associated with every node
  mark1 = net.nin;
  mark2 = mark1 + 1;
  mark3 = mark2 + net.nhidden;
  mark4 = mark3 + 1;
  mark5 = mark4 + net.nout;
  a_alphaI = a_alpha(1:mark1);
  a_alphaIb = a_alpha(mark1+1:mark2);
  a_alphaH = a_alpha(mark2+1:mark3);
  a_alphaHb = a_alpha((mark3+1):mark4);
  a_alphaO = a_alpha((mark4+1):mark5);
  b_alphaI = b_alpha(1:mark1);
  b_alphaIb = b_alpha(mark1+1:mark2);
  b_alphaH = b_alpha(mark2+1:mark3);
  b_alphaHb = b_alpha((mark3+1):mark4);
  b_alphaO = b_alpha((mark4+1):mark5);
  abar_alphaI = a_alphaI + net.nhidden/2;
  abar_alphaIb = a_alphaIb + net.nhidden/2;
  abar_alphaH = a_alphaH(1:net.nhidden, 1) + ...
      (net.nin+1)/2 + net.nout/2;
  abar_alphaHb = a_alphaHb + net.nout/2;
  abar_alphaO = a_alphaO + (net.nhidden+1)/2;
  abar_alpha = [abar_alphaI; abar_alphaIb; ...
      abar_alphaH; abar_alphaHb; ...
      abar_alphaO];
  
 % alpha = abar_alpha./net.alphaposterior.b; 
 % alphaI = alpha(1:mark1);
 % alphaIb = alpha(mark1+1:mark2);
 % alphaH = alpha(mark2+1:mark3);
 % alphaHb = alpha(mark3+1:mark4);
 % alphaO = alpha(mark4+1:mark5);

  bbar_alphaI = b_alphaI + sqrt(sum(sum(exp2Net.w1, 2), 1)/2); 
  alphaI = abar_alphaI./bbar_alphaI;
  bbar_alphaIb = b_alphaIb + sqrt(sum(exp2Net.b1, 2)/2); 
  alphaIb = abar_alphaIb./bbar_alphaIb;
  bbar_alphaO = b_alphaO + sqrt(sum(sum(exp2Net.w2, 2), 1)/2); 
  alphaO = abar_alphaO./bbar_alphaO;
  bbar_alphaH = b_alphaH ...
      + exp2Net.w1'/2*alphaI ...
      + exp2Net.b1'/2*alphaIb ...
      + exp2Net.w2/2*alphaO;
  alphaH = abar_alphaH./bbar_alphaH;
  bbar_alphaHb = b_alphaHb + ...
      exp2Net.b2/2*alphaO;
  alphaHb = abar_alphaHb./bbar_alphaHb;


  oldbbar_alpha = [bbar_alphaI; bbar_alphaIb; ...
		   bbar_alphaH; bbar_alphaHb; ...
		   bbar_alphaO];
  bbar_alpha = zeros(size(oldbbar_alpha));
  count = 0;
  while (sum(abs(oldbbar_alpha-bbar_alpha))>1e-4 & count <=10000)
    count = count + 1;
    oldbbar_alpha = bbar_alpha;
    bbar_alphaI = b_alphaI ...
	+ exp2Net.w1/2*alphaH; 
    alphaI = abar_alphaI./bbar_alphaI;
    bbar_alphaIb = b_alphaIb ...
	+ exp2Net.b1/2*alphaH;
    alphaIb = abar_alphaIb./bbar_alphaIb;
    bbar_alphaH = b_alphaH ...
	+ exp2Net.w1'/2*alphaI ...
	+ exp2Net.b1'/2*alphaIb ...
	+ exp2Net.w2/2*alphaO;
    alphaH = abar_alphaH./bbar_alphaH;
    bbar_alphaHb = b_alphaHb + ...
	exp2Net.b2/2*alphaO;
    alphaHb = abar_alphaHb./bbar_alphaHb;
    bbar_alphaO = b_alphaO ...
	+ exp2Net.w2'/2*alphaH ...
	+ exp2Net.b2'/2*alphaHb;
    alphaO = abar_alphaO./bbar_alphaO;
    oldbbar_alpha = bbar_alpha;
    bbar_alpha = [bbar_alphaI; bbar_alphaIb; ...
		  bbar_alphaH; bbar_alphaHb; ...
		  bbar_alphaO];
  end  
  disp(count)
  if count>10000
    warning('Maximum iterations exceeded in hyperparameter update')
  end
end

net.alphaposterior.a = abar_alpha;
net.alphaposterior.b = bbar_alpha;

a_beta = net.betaprior.a;
b_beta = net.betaprior.b;

ndata = size(x, 1);
switch net.covstrct
case 'none'
  [sexp, qexp] = ensoutputexpec(net, x, 'hess', x, t);

otherwise
  [sexp, qexp] = ensoutputexpec(net, x);
end
vector = (qexp - sexp.^2)<0;
qexp = qexp.*(1-vector)+(vector.*sexp.^2);
edata = 0.5*sum(qexp - 2*t.*sexp + t.*t, 1);

net.betaposterior.a = a_beta + ndata/2;
net.betaposterior.b = b_beta + edata;













