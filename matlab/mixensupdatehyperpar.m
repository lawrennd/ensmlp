function net = mixensupdatehyperpar(net, x, t)

% MIXENSUPDATEHYPERPAR Re-estimate parameters of the hyper posteriors.
% FORMAT
% DESC re-estimates the hyperparameters ALPHA and BETA. 
% ARG net : the network to estimate the hyperparameters for.
% ARG x : input data locations.
% ARG t : target data locations.
% RETURN net : the network with updated hyperparameters.
%
% SEEALSO : MIXENSHYPERPRIOR, MIXENSGRAD, MIXENSERR
%
% COPYRIGHT : Neil D Lawrence and Mehdi Azzouzi, 1999
  
% ENSMLP


a_alpha = net.ens(1).alphaprior.a;
b_alpha = net.ens(1).alphaprior.b;

exp2Net = net.ens(1);

% Check the type of softmax
if strcmp(net.soft, 'y') == 1
  mix_coeff = get_pi(net.z);
else
  mix_coeff = net.pi;
end

switch net.ens(1).covstrct
 case 'none'
  error('Not implemented')
 case 'diag'    
  exp2Net.w1 = zeros(size(exp2Net.w1));
  exp2Net.b1 = zeros(size(exp2Net.b1));
  exp2Net.w2 = zeros(size(exp2Net.w2));
  exp2Net.b2 = zeros(size(exp2Net.b2));
  for m=1:net.M 
    exp2Net.w1 = exp2Net.w1 + mix_coeff(m)*(net.ens(m).d1.^2 ...
					    + net.ens(m).w1.^2);
    exp2Net.b1 = exp2Net.b1 + mix_coeff(m)*(net.ens(m).db1.^2 ...
					    + net.ens(m).b1.^2);
    exp2Net.w2 = exp2Net.w2 + mix_coeff(m)*(net.ens(m).d2.^2 ...
					  + net.ens(m).w2.^2);
    exp2Net.b2 = exp2Net.b2 + mix_coeff(m)*(net.ens(m).db2.^2 ...
					    + net.ens(m).b2.^2);
  end
  
 otherwise
  error('Covariance function not yet implemented')
end 

 
switch net.ens(1).alphaprior.type
 case 'single'
  % hyper prior is single value
  bbar_alpha = b_alpha;
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
  % Automatic relevance prior
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
  bbar_alphaI = b_alphaI;
  bbar_alphaIb = b_alphaIb;
  bbar_alphaH = b_alphaH;
  bbar_alphaHb = b_alphaHb;
  

  bbar_alphaI = b_alphaI + sum(exp2Net.w1, 2)/2; 
  bbar_alphaIb = b_alphaIb + sum(exp2Net.b1, 2)/2; 
  bbar_alphaH = b_alphaH + sum(sum(exp2Net.w2, 2), 2)/2; 
  bbar_alphaHb = b_alphaHb + sum(exp2Net.b2, 2)/2; 
  abar_alpha = [abar_alphaI; abar_alphaIb; ...
		abar_alphaH; abar_alphaHb];
  bbar_alpha = [bbar_alphaI; bbar_alphaIb; ...
		bbar_alphaH; bbar_alphaHb];

 case 'prune'
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
  
  %  alpha = abar_alpha./net.ens(1).alphaposterior.b; 
  %  alphaI = alpha(1:mark1);
  %  alphaIb = alpha(mark1+1:mark2);
  %  alphaH = alpha(mark2+1:mark3);
  %  alphaHb = alpha(mark3+1:mark4);
  %  alphaO = alpha(mark4+1:mark5);
  
  %  oldbbar_alpha = zeros(size(net.ens(1).alphaposterior.b));
  %  bbar_alpha = net.ens(1).alphaposterior.b;
  %  count = 0;
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
    bbar_alphaO = b_alphaO ...
	+ exp2Net.w2'/2*alphaH ...
	+ exp2Net.b2'/2*alphaHb;
    alphaO = abar_alphaO./bbar_alphaO;
    bbar_alphaH = b_alphaH ...
	+ exp2Net.w1'/2*alphaI ...
	+ exp2Net.b1'/2*alphaIb ...
	+ exp2Net.w2/2*alphaO;
    alphaH = abar_alphaH./bbar_alphaH;
    bbar_alphaHb = b_alphaHb + ...
	exp2Net.b2/2*alphaO;
    alphaHb = abar_alphaHb./bbar_alphaHb;
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
for m = 1:net.M
  net.ens(m).alphaposterior.a = abar_alpha;
  net.ens(m).alphaposterior.b = bbar_alpha;
end
a_beta = net.ens(1).betaprior.a;
b_beta = net.ens(1).betaprior.b;

ndata = size(x, 1);
[sexp, qexp] = mixensoutputexpec(net, x);
vector = (qexp - sexp.^2)<0;
qexp = qexp.*(1-vector)+(vector.*sexp.^2);
edata = 0.5*sum(qexp - 2*t.*sexp + t.*t, 1);

for m = 1:net.M
  net.ens(m).betaposterior.a = a_beta + ndata/2;
  net.ens(m).betaposterior.b = b_beta + edata;
end



























